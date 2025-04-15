from src.prompting.renderedPromptRecord import RenderedPromptRecord
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
    records  = [RenderedPromptRecord.load_from_file_static(file) for file in records_path]
    dfs = pd.DataFrame()
    for record in records:
        df = pd.DataFrame(record.get_merged_data())
        dfs = pd.concat([dfs, df], ignore_index=True)
    
    dfs.drop(["message"], axis=1, inplace=True)

    dfs["messageId"] = dfs["messageId"].astype(str)
    dfs["responseId"] = dfs["responseId"].astype(str)
    analysis_df = pd.json_normalize(df['trait'])
    dfs = pd.concat([dfs, analysis_df], axis=1)
    dfs.drop(columns=['trait',"message_text"], inplace=True)
    dfs.to_parquet(
        os.path.join(record_folder, "merged_data.parquet"),
        index=False,
    )

if __name__ == '__main__':
    absl.app.run(main)
