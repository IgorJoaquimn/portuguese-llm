from src.prompting.renderedPromptRecord import RenderedPromptRecord
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import absl.flags
import absl.app
import os

FLAGS = absl.flags.FLAGS

# Definição das flags
absl.flags.DEFINE_string("record_folder", None, "Path that contains the desired record")
absl.flags.mark_flag_as_required("record_folder")
def main(_):
    record_folder = FLAGS.record_folder
    records_path = [record_folder + f for f in os.listdir(record_folder) if ".pickle" in f]
    records  = [RenderedPromptRecord.load_from_file_static(file) for file in tqdm(records_path, desc="Loading records")]
    dfs = pd.DataFrame()
    for record in tqdm(records, desc="Processing records"):
        df = pd.DataFrame(record.get_merged_data())
        dfs = pd.concat([dfs, df], ignore_index=True)
    analysis_df = pd.json_normalize(dfs['trait'])
    dfs = pd.concat([dfs, analysis_df], axis=1)

    # Convert object columns to string
    for col in dfs.columns:
        if dfs[col].dtype == 'object':
            # Handle potential lists or other non-scalar data by converting to string
            dfs[col] = dfs[col].astype(str)

    output_path = os.path.join(record_folder, "merged_data.parquet")
    print(f"\nSaving merged data to {output_path}")

    dfs.to_parquet(
        output_path,
        index=False,
    )

    print("Successfully saved the Parquet file.")
    print(dfs.info())

if __name__ == '__main__':
    absl.app.run(main)
