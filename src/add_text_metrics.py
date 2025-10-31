#!/usr/bin/env python3
"""
Unified Text Metrics Enrichment Script

Reads a parquet file, computes comprehensive text metrics, and outputs an enriched 
parquet file with all metrics appended as columns.

Metrics included:
- Lemmatization (lemmas column)
- Linguistic complexity (MLC, MLS, DCC, CPC, tree depths, TTR, lexical density)
- Stylometric analysis (Flesch Reading Ease, POS frequencies, NER, lexical metrics)

Usage:
    python -m src.add_text_metrics \
        --input_file data/merged_data.parquet \
        --output_file data/merged_data_with_metrics.parquet \
        --text_column response \
        --batch_size 100
"""

import sys
import os
import argparse
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import signal
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.text_metrics import TextMetricsAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParquetMetricsEnricher:
    """Enriches parquet files with computed text metrics with checkpointing support."""
    
    def __init__(self, batch_size: int = 100, n_workers: int = None, parallel_method: str = "sequential",
                 checkpoint_interval: int = 100, checkpoint_dir: str = None):
        """
        Initialize the enricher.
        
        Args:
            batch_size: Number of rows to process at once
            n_workers: Number of parallel workers (default: min(4, CPU count))
            parallel_method: "sequential", "thread", or "process"
            checkpoint_interval: Save progress every N rows (0 to disable)
            checkpoint_dir: Directory to save checkpoints (default: .checkpoints)
        """
        self.batch_size = batch_size
        self.metrics_analyzer = None
        self.processed_count = 0
        self.failed_count = 0
        self.n_workers = n_workers or min(4, cpu_count())
        self.parallel_method = parallel_method
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir or ".checkpoints"
        self.metrics_list = None
        self.current_job_id = None
    
    def _get_checkpoint_path(self, job_id: str) -> str:
        """Get checkpoint file path."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, f"{job_id}_metrics.pkl")
    
    def _get_metadata_path(self, job_id: str) -> str:
        """Get metadata file path."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, f"{job_id}_metadata.json")
    
    def _save_checkpoint(self, job_id: str, metrics_list: list, metadata: dict) -> None:
        """Save checkpoint data."""
        if self.checkpoint_interval == 0:
            return
        
        try:
            import pickle
            checkpoint_path = self._get_checkpoint_path(job_id)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(metrics_list, f)
            
            metadata_path = self._get_metadata_path(job_id)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"✓ Checkpoint saved: {len(metrics_list)} rows, {metadata['processed']} processed")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, job_id: str) -> tuple:
        """Load checkpoint data. Returns (metrics_list, metadata) or (None, None) if not found."""
        try:
            import pickle
            checkpoint_path = self._get_checkpoint_path(job_id)
            metadata_path = self._get_metadata_path(job_id)
            
            if not os.path.exists(checkpoint_path) or not os.path.exists(metadata_path):
                return None, None
            
            with open(checkpoint_path, 'rb') as f:
                metrics_list = pickle.load(f)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"✓ Checkpoint loaded: {len(metrics_list)} rows, {metadata['processed']} already processed")
            return metrics_list, metadata
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None, None
    
    def _cleanup_checkpoint(self, job_id: str) -> None:
        """Delete checkpoint files after successful completion."""
        try:
            import pickle
            checkpoint_path = self._get_checkpoint_path(job_id)
            metadata_path = self._get_metadata_path(job_id)
            
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            logger.info(f"✓ Checkpoint cleaned up: {job_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")
        
    def initialize_analyzer(self) -> bool:
        """Initialize the metrics analyzer."""
        try:
            logger.info("Initializing TextMetricsAnalyzer...")
            self.metrics_analyzer = TextMetricsAnalyzer(udpipe_enabled=True)
            logger.info("✓ Metrics analyzer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize metrics analyzer: {e}")
            return False
    
    def enrich_parquet(
        self,
        input_file: str,
        output_file: str,
        text_column: str = "response",
        udpipe_column: str = "udpipe_result",
        resume: bool = True
    ) -> bool:
        """
        Load parquet, compute metrics, and save enriched parquet.
        
        Args:
            input_file: Path to input parquet file
            output_file: Path to output parquet file
            text_column: Column with text to analyze
            udpipe_column: Column with UDPipe results
            resume: Whether to resume from checkpoint if available
        """
        try:
            # Create job ID based on input/output files
            job_id = f"{Path(input_file).stem}_{int(time.time())}"
            self.current_job_id = job_id
            
            # Load data
            logger.info(f"Loading parquet file from: {input_file}")
            df = pd.read_parquet(input_file)
            logger.info(f"✓ Loaded {len(df)} rows")
            
            # Validate columns
            if text_column not in df.columns:
                logger.error(f"Column '{text_column}' not found in parquet file")
                return False
            
            has_udpipe = udpipe_column in df.columns
            if has_udpipe:
                logger.info(f"✓ Found UDPipe column: {udpipe_column}")
            else:
                logger.warning(f"Column '{udpipe_column}' not found. Will compute UDPipe metrics on demand.")
                df[udpipe_column] = ""
            
            # Try to load checkpoint if resume is enabled
            metrics_list = None
            checkpoint_metadata = None
            if resume and self.checkpoint_interval > 0:
                metrics_list, checkpoint_metadata = self._load_checkpoint(job_id)
                if metrics_list is not None:
                    self.processed_count = checkpoint_metadata.get('processed', 0)
                    self.failed_count = checkpoint_metadata.get('failed', 0)
                    logger.info(f"Resuming from checkpoint: {self.processed_count} processed, {self.failed_count} failed")
            
            # Compute metrics (naturally handles both fresh and resume scenarios)
            logger.info(f"Computing metrics for {len(df)} rows...")
            metrics_list = self._compute_metrics(df, text_column, udpipe_column, existing_metrics=metrics_list)
            
            if metrics_list is None:
                return False
            
            # Add metrics to dataframe
            logger.info(f"Adding metrics as new columns...")
            metrics_df = pd.DataFrame(metrics_list)
            
            # Remove duplicate columns if they already exist in df
            duplicate_cols = [col for col in metrics_df.columns if col in df.columns]
            if duplicate_cols:
                logger.warning(f"Removing duplicate columns that already exist: {duplicate_cols}")
                metrics_df = metrics_df.drop(columns=duplicate_cols)
            
            df_enriched = pd.concat([df, metrics_df], axis=1)
            
            logger.info(f"✓ Added {len(metrics_df.columns)} metric columns")
            logger.info(f"✓ Processed: {self.processed_count} rows successfully")
            if self.failed_count > 0:
                logger.warning(f"⚠ Failed: {self.failed_count} rows")
            
            # Save enriched parquet
            logger.info(f"Saving enriched parquet to: {output_file}")
            df_enriched.to_parquet(output_file, index=False)
            logger.info(f"✓ Successfully saved {len(df_enriched)} rows")
            
            # Clean up checkpoint on success
            self._cleanup_checkpoint(job_id)
            
            self._print_summary(df, df_enriched, metrics_df)
            return True
            
        except KeyboardInterrupt:
            logger.warning(f"\n⚠ Process interrupted! Your progress has been saved.")
            logger.warning(f"   Job ID: {job_id}")
            logger.warning(f"   To resume, run: python -m src.add_text_metrics ... (same command)")
            return False
            
        except Exception as e:
            logger.error(f"Error during enrichment: {e}")
            logger.warning(f"⚠ Progress saved to checkpoint for recovery.")
            logger.warning(f"   Job ID: {job_id}")
            return False
    
    def _compute_metrics(self, df: pd.DataFrame, text_column: str, udpipe_column: str = None,
                        existing_metrics: list = None) -> list:
        """Compute metrics for all rows using configured parallel method, optionally resuming from checkpoint."""
        # Initialize or resume metrics list
        metrics_list = existing_metrics if existing_metrics is not None else [None] * len(df)
        
        # Find starting index (first None entry if resuming)
        start_idx = 0
        if existing_metrics is not None:
            for i, metrics in enumerate(existing_metrics):
                if metrics is None:
                    start_idx = i
                    break
            else:
                # All rows already computed
                logger.info("All rows already processed from checkpoint")
                return metrics_list
        
        if start_idx > 0:
            logger.info(f"Resuming from row {start_idx}")
        
        # Use appropriate execution strategy
        if self.parallel_method == "sequential":
            self._execute_sequential(df, text_column, udpipe_column, metrics_list, start_idx)
        elif self.parallel_method == "thread":
            self._execute_parallel(df, text_column, udpipe_column, metrics_list, start_idx, ThreadPoolExecutor)
        elif self.parallel_method == "process":
            self._execute_parallel(df, text_column, udpipe_column, metrics_list, start_idx, ProcessPoolExecutor)
        else:
            logger.warning(f"Unknown parallel method: {self.parallel_method}. Using sequential.")
            self._execute_sequential(df, text_column, udpipe_column, metrics_list, start_idx)
        
        return metrics_list
    
    def _execute_sequential(self, df: pd.DataFrame, text_column: str, udpipe_column: str,
                           metrics_list: list, start_idx: int) -> None:
        """Execute sequential computation, processing rows from start_idx onward."""
        rows_to_process = range(start_idx, len(df))
        
        for idx in tqdm(rows_to_process, desc="Computing metrics", initial=start_idx, total=len(df)):
            if metrics_list[idx] is not None:
                continue  # Skip already computed
            
            row = df.iloc[idx]
            text = row.get(text_column, '')
            udpipe_output = row.get(udpipe_column, '') if udpipe_column else None
            
            metrics_list[idx] = self._process_row(idx, text, udpipe_output)
            
            # Save checkpoint periodically
            if self.checkpoint_interval > 0 and (idx - start_idx + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(self.current_job_id, metrics_list, {
                    "last_idx": idx,
                    "processed": self.processed_count,
                    "failed": self.failed_count
                })
    
    def _execute_parallel(self, df: pd.DataFrame, text_column: str, udpipe_column: str,
                         metrics_list: list, start_idx: int, executor_class) -> None:
        """Execute parallel computation, processing rows from start_idx onward."""
        # Prepare rows to process
        rows_to_process = [
            (idx, df.iloc[idx].get(text_column, ''), df.iloc[idx].get(udpipe_column, '') if udpipe_column else None)
            for idx in range(start_idx, len(df))
            if metrics_list[idx] is None
        ]
        
        if not rows_to_process:
            return
        
        logger.info(f"Using {self.parallel_method} processing with {self.n_workers} workers")
        
        try:
            with executor_class(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(self._analyze_row, idx, text, udpipe_output): idx
                    for idx, text, udpipe_output in rows_to_process
                }
                
                # Process results as they complete
                processed_since_checkpoint = 0
                with tqdm(total=len(rows_to_process), desc="Computing metrics") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            metrics = future.result()
                            metrics_list[idx] = metrics
                            if metrics is not None and metrics != self.metrics_analyzer._get_empty_metrics():
                                self.processed_count += 1
                            else:
                                self.failed_count += 1
                        except Exception as e:
                            logger.debug(f"Error analyzing row {idx}: {e}")
                            metrics_list[idx] = self.metrics_analyzer._get_empty_metrics()
                            self.failed_count += 1
                        
                        processed_since_checkpoint += 1
                        pbar.update(1)
                        
                        # Save checkpoint periodically
                        if self.checkpoint_interval > 0 and processed_since_checkpoint % self.checkpoint_interval == 0:
                            self._save_checkpoint(self.current_job_id, metrics_list, {
                                "processed": self.processed_count,
                                "failed": self.failed_count
                            })
        
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            # Continue (checkpoint already saved)
    
    def _process_row(self, idx: int, text: str, udpipe_output: str = None) -> dict:
        """Process a single row and return metrics dict."""
        # Skip empty text
        if not text or (isinstance(text, float) and pd.isna(text)):
            self.failed_count += 1
            return self.metrics_analyzer._get_empty_metrics()
        
        text = str(text) if text is not None else ''
        udpipe_output = str(udpipe_output) if udpipe_output else None
        
        if udpipe_output and (not udpipe_output or udpipe_output == 'nan' or udpipe_output == ''):
            udpipe_output = None
        
        try:
            metrics = self.metrics_analyzer.analyze(
                text,
                udpipe_output=udpipe_output,
                include_ner=True,
                include_lemmatization=True
            )
            self.processed_count += 1
            return metrics
        except Exception as e:
            logger.debug(f"Error analyzing row {idx}: {e}")
            self.failed_count += 1
            return self.metrics_analyzer._get_empty_metrics()

    def _analyze_row(self, idx: int, text: str, udpipe_output: str = None):
        """Analyze a single row (helper for parallel processing)."""
        # Skip empty text
        if not text or (isinstance(text, float) and pd.isna(text)):
            return self.metrics_analyzer._get_empty_metrics()
        
        text = str(text) if text is not None else ''
        udpipe_output = str(udpipe_output) if udpipe_output else None
        
        if udpipe_output and (not udpipe_output or udpipe_output == 'nan' or udpipe_output == ''):
            udpipe_output = None
        
        try:
            metrics = self.metrics_analyzer.analyze(
                text,
                udpipe_output=udpipe_output,
                include_ner=True,
                include_lemmatization=True
            )
            return metrics
        except Exception as e:
            logger.debug(f"Error analyzing row {idx}: {e}")
            return self.metrics_analyzer._get_empty_metrics()
    
    def _print_summary(self, df_original: pd.DataFrame, df_enriched: pd.DataFrame, metrics_df: pd.DataFrame):
        """Print summary of enrichment operation."""
        print("\n" + "=" * 80)
        print("📊 ENRICHMENT SUMMARY")
        print("=" * 80)
        print(f"Original rows:        {len(df_original)}")
        print(f"Enriched rows:        {len(df_enriched)}")
        print(f"Original columns:     {len(df_original.columns)}")
        print(f"New columns added:    {len(metrics_df.columns)}")
        print(f"Total columns:        {len(df_enriched.columns)}")
        print()
        print("📈 New Metric Columns:")
        for col in sorted(metrics_df.columns):
            print(f"  • {col}")
        print()
        print(f"✅ Processing stats:  {self.processed_count} successful, {self.failed_count} failed")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich parquet file with computed text metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential processing (default, slowest but most stable)
  python -m src.add_text_metrics --input_file data/input.parquet --output_file data/output.parquet

  # Parallel with threads (faster, good for I/O bound tasks)
  python -m src.add_text_metrics --input_file data/input.parquet --output_file data/output.parquet --parallel thread --workers 8

  # Parallel with processes (faster for CPU-intensive tasks)
  python -m src.add_text_metrics --input_file data/input.parquet --output_file data/output.parquet --parallel process --workers 4
        """
    )
    
    parser.add_argument("--input_file", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output parquet file")
    parser.add_argument("--text_column", type=str, default="response", help="Text column name (default: response)")
    parser.add_argument("--udpipe_column", type=str, default="udpipe_result", help="UDPipe column name (default: udpipe_result)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size (default: 100)")
    parser.add_argument("--parallel", type=str, choices=["sequential", "thread", "process"], default="process",
                        help="Processing method (default: process)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers for parallel processing (default: min(4, CPU count))")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save checkpoint every N rows (default: 100, set to 0 to disable)")
    parser.add_argument("--checkpoint_dir", type=str, default=".checkpoints",
                        help="Directory for checkpoint files (default: .checkpoints)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable checkpoint resuming (start fresh even if checkpoint exists)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    enricher = ParquetMetricsEnricher(
        batch_size=args.batch_size,
        n_workers=args.workers,
        parallel_method=args.parallel,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    if not enricher.initialize_analyzer():
        return 1
    
    if enricher.enrich_parquet(args.input_file, args.output_file, args.text_column, args.udpipe_column, 
                               resume=not args.no_resume):
        logger.info("✅ Enrichment completed successfully!")
        return 0
    else:
        logger.error("❌ Enrichment failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
