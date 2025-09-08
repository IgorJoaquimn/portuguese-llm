#!/usr/bin/env python3
"""
Lemmatization script using UDPipe for Portuguese text data.
Enhanced with robust error handling and progress tracking.
"""

import sys
import json
import signal
import logging
import time
import os
import pandas as pd
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.udpipe.udpipe_utils import UDPipeClient, extract_lemmas_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global function for multiprocessing (needs to be picklable)
def lemmatize_text_worker(text_data):
    """Worker function for multiprocessing - must be at module level to be picklable."""
    idx, text = text_data
    try:
        if not text or pd.isna(text) or text.strip() == "":
            return idx, ""
        
        # Create a new client for each worker
        client = UDPipeClient()
        
        # Get UDPipe response
        udpipe_output = client.generate_one_response(text)
        
        # Parse response
        sentences = client.parse_response(udpipe_output)
        
        # Extract lemmas
        lemmas = extract_lemmas_string(sentences)
        
        return idx, lemmas
        
    except Exception as e:
        error_info = {
            "row_index": idx,
            "text_preview": text[:100] if text else "None",
            "error": str(e),
            "timestamp": time.time()
        }
        
        return idx, "", error_info  # Return error info as well


class LemmatizationProcessor:
    """Process DataFrame text columns for lemmatization using UDPipe."""
    
    def __init__(self, save_interval=100, n_workers=None, parallel_method="thread"):
        self.udpipe_client = UDPipeClient()
        self.save_interval = save_interval
        self.processed_count = 0
        self.failed_items = []
        self.current_df = None
        self.n_workers = n_workers or min(8, cpu_count())  # Default to 8 workers or CPU count
        self.parallel_method = parallel_method  # "thread", "process", or "sequential"
        
        # Only use lock for threading (not needed for multiprocessing)
        if parallel_method == "thread":
            self.lock = threading.Lock()
        else:
            self.lock = None
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.handle_sigint)
        signal.signal(signal.SIGINT, self.handle_sigint)
    
    def lemmatize_text_safe(self, text_data):
        """Thread-safe wrapper for lemmatizing text with error handling (for threading only)."""
        idx, text = text_data
        try:
            if not text or pd.isna(text) or text.strip() == "":
                return idx, ""
            
            # Create a new client for threading to avoid conflicts
            client = UDPipeClient()
            
            # Get UDPipe response
            udpipe_output = client.generate_one_response(text)
            
            # Parse response
            sentences = client.parse_response(udpipe_output)
            
            # Extract lemmas
            lemmas = extract_lemmas_string(sentences)
            
            return idx, lemmas
            
        except Exception as e:
            error_info = {
                "row_index": idx,
                "text_preview": text[:100] if text else "None",
                "error": str(e),
                "timestamp": time.time()
            }
            
            if self.lock:
                with self.lock:
                    self.failed_items.append(error_info)
            else:
                self.failed_items.append(error_info)
            
            return idx, ""  # Return empty string on failure
    
    def process_dataframe_parallel(self, df, text_column="response", output_column="response_lemm"):
        """Process DataFrame with parallel processing."""
        self.current_df = df.copy()
        total_rows = len(df)
        
        # Initialize output column if it doesn't exist
        if output_column not in self.current_df.columns:
            self.current_df[output_column] = ""
        
        # Filter rows that need processing (skip rows that already have lemmatized text)
        mask_empty = (
            self.current_df[output_column].isna() | 
            (self.current_df[output_column] == "") | 
            (self.current_df[output_column] == "None")
        )
        rows_to_process = self.current_df[mask_empty]
        rows_already_processed = total_rows - len(rows_to_process)
        
        logger.info(f"Starting parallel lemmatization of {len(rows_to_process)} rows (skipping {rows_already_processed} already processed)")
        logger.info(f"Text column: {text_column}")
        logger.info(f"Output column: {output_column}")
        logger.info(f"Workers: {self.n_workers}")
        logger.info(f"Parallel method: {self.parallel_method}")
        
        # If no rows to process, return current df
        if len(rows_to_process) == 0:
            logger.info("No rows need processing. All rows already have lemmatized text.")
            return self.current_df
        
        # Prepare data for parallel processing (only rows that need processing)
        text_data = [(idx, row[text_column]) for idx, row in rows_to_process.iterrows()]
        
        try:
            if self.parallel_method == "thread":
                executor_class = ThreadPoolExecutor
                worker_func = self.lemmatize_text_safe
            elif self.parallel_method == "process":
                executor_class = ProcessPoolExecutor
                worker_func = lemmatize_text_worker  # Use global function for processes
            else:
                # Sequential processing
                return self.process_dataframe_sequential(df, text_column, output_column)
            
            with executor_class(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(worker_func, data): data[0] 
                    for data in text_data
                }
                
                # Process results as they complete
                with tqdm(total=len(rows_to_process), desc="Lemmatizing") as pbar:
                    for future in as_completed(future_to_idx):
                        try:
                            result = future.result()
                            
                            # Handle different return formats
                            if len(result) == 3:  # Process result with error
                                idx, lemmatized, error_info = result
                                if error_info:
                                    self.failed_items.append(error_info)
                            else:  # Thread result
                                idx, lemmatized = result
                            
                            self.current_df.at[idx, output_column] = lemmatized
                            
                            if self.lock:
                                with self.lock:
                                    self.processed_count += 1
                                    if self.processed_count % self.save_interval == 0:
                                        logger.info(f"Progress: {self.processed_count}/{len(rows_to_process)} processed (total: {self.processed_count + rows_already_processed}/{total_rows})")
                            else:
                                self.processed_count += 1
                                if self.processed_count % self.save_interval == 0:
                                    logger.info(f"Progress: {self.processed_count}/{len(rows_to_process)} processed (total: {self.processed_count + rows_already_processed}/{total_rows})")
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"Error processing future: {e}")
                            pbar.update(1)
                            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Saving current progress...")
            self.handle_sigint(signal.SIGINT, None)
            
        except Exception as e:
            logger.error(f"Critical error during parallel processing: {e}")
            raise e
            
        finally:
            self.save_failed_items()
            logger.info(f"\n=== Processing Summary ===")
            logger.info(f"Total rows: {total_rows}")
            logger.info(f"Successfully processed: {self.processed_count}")
            logger.info(f"Failed items: {len(self.failed_items)}")
            
        return self.current_df
    
    def process_dataframe_sequential(self, df, text_column="response", output_column="response_lemm"):
        """Sequential processing (original method)."""
        self.current_df = df.copy()
        total_rows = len(df)
        
        # Initialize output column if it doesn't exist
        if output_column not in self.current_df.columns:
            self.current_df[output_column] = ""
        
        # Filter rows that need processing (skip rows that already have lemmatized text)
        mask_empty = (
            self.current_df[output_column].isna() | 
            (self.current_df[output_column] == "") | 
            (self.current_df[output_column] == "None")
        )
        rows_to_process = self.current_df[mask_empty]
        rows_already_processed = total_rows - len(rows_to_process)
        
        logger.info(f"Starting sequential lemmatization of {len(rows_to_process)} rows (skipping {rows_already_processed} already processed)")
        logger.info(f"Text column: {text_column}")
        logger.info(f"Output column: {output_column}")
        
        # If no rows to process, return current df
        if len(rows_to_process) == 0:
            logger.info("No rows need processing. All rows already have lemmatized text.")
            return self.current_df
        
        try:
            # Process with progress bar (only rows that need processing)
            for idx, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Lemmatizing"):
                text = row[text_column]
                
                try:
                    idx, lemmatized = self.lemmatize_text_safe((idx, text))
                    self.current_df.at[idx, output_column] = lemmatized
                    self.processed_count += 1
                    
                    # Periodic save
                    if self.processed_count % self.save_interval == 0:
                        logger.info(f"Progress: {self.processed_count}/{len(rows_to_process)} processed (total: {self.processed_count + rows_already_processed}/{total_rows})")
                        
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    self.current_df.at[idx, output_column] = ""
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Saving current progress...")
            self.handle_sigint(signal.SIGINT, None)
            
        except Exception as e:
            logger.error(f"Critical error during processing: {e}")
            raise e
            
        finally:
            self.save_failed_items()
            logger.info(f"\n=== Processing Summary ===")
            logger.info(f"Total rows: {total_rows}")
            logger.info(f"Rows already processed (skipped): {rows_already_processed}")
            logger.info(f"Rows processed in this run: {self.processed_count}")
            logger.info(f"Successfully processed: {self.processed_count + rows_already_processed}")
            logger.info(f"Failed items: {len(self.failed_items)}")
            
        return self.current_df
    
    def save_failed_items(self, output_path=None):
        """Save failed items to JSON file."""
        if self.failed_items:
            try:
                if output_path is None:
                    output_path = "lemmatization_failed_items.json"
                
                with open(output_path, 'w') as f:
                    json.dump(self.failed_items, f, indent=2)
                logger.info(f"Failed items saved to: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to save failed items: {e}")
    
    def handle_sigint(self, signal_received, frame):
        signal_str = signal.Signals(signal_received).name if signal_received else "UNKNOWN"
        logger.info(f"\n{signal_str} detected! Running cleanup function...")
        
        try:
            if self.current_df is not None:
                emergency_path = f"lemmatization_emergency_save_{int(time.time())}.parquet"
                self.current_df.to_parquet(emergency_path, index=False)
                logger.info(f"Emergency save completed: {emergency_path}")
                logger.info(f"Processed {self.processed_count} items before interruption.")
                
            self.save_failed_items("lemmatization_failed_items_emergency.json")
            
        except Exception as e:
            logger.error(f"Error during emergency save: {e}")
            
        logger.info("Cleanup completed. Exiting gracefully...")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Lemmatize text data using UDPipe")
    
    parser.add_argument("--input_file", type=str, default="data/merged_data.parquet",
                        help="Path to input parquet file")
    parser.add_argument("--output_file", type=str, default="data/merged_data_lemm.parquet",
                        help="Path to output parquet file")
    parser.add_argument("--text_column", type=str, default="response",
                        help="Column name containing text to lemmatize")
    parser.add_argument("--output_column", type=str, default="response_lemm",
                        help="Column name for lemmatized output")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save progress every N items")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: min(8, CPU count))")
    parser.add_argument("--parallel", type=str, default="thread",
                        choices=["thread", "process", "sequential"],
                        help="Parallel processing method")
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from: {args.input_file}")
        df = pd.read_parquet(args.input_file)
        logger.info(f"Loaded {len(df)} rows")
        
        # Check if text column exists
        if args.text_column not in df.columns:
            raise ValueError(f"Text column '{args.text_column}' not found in data")
        
        # Process lemmatization
        processor = LemmatizationProcessor(
            save_interval=args.save_interval,
            n_workers=args.workers,
            parallel_method=args.parallel
        )
        
        if args.parallel == "sequential":
            result_df = processor.process_dataframe_sequential(
                df, 
                text_column=args.text_column,
                output_column=args.output_column
            )
        else:
            result_df = processor.process_dataframe_parallel(
                df, 
                text_column=args.text_column,
                output_column=args.output_column
            )
        
        # Save results
        logger.info(f"Saving results to: {args.output_file}")
        result_df.to_parquet(args.output_file, index=False)
        logger.info("Lemmatization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        raise e


if __name__ == "__main__":
    main()
