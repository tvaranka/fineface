import pandas as pd
import datasets
from pathlib import Path
from typing import List, Dict
import os
import numpy as np
import rarfile
import PIL
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
from tqdm import tqdm

class RARImage(datasets.Image):
    def __init__(self, rar_file = None):
        super().__init__()
        self.rf = rar_file
        
    def decode_example(self, value: dict, token_per_repo_id=None) -> PIL.Image.Image:
        path = value["path"]
        if "Manually" in path: #Affectnet RAR samples
            bytes_ = self.rf.read(path)
            image = PIL.Image.open(BytesIO(bytes_))
        else: # OTHER
            image = PIL.Image.open(path)
        image.load()  # to avoid "Too many open files" errors
        return image

def _normalize_to_range(arr, new_max=1):
    current_min = np.min(arr)
    current_max = np.max(arr)
    normalized_arr = ((arr - current_min) / (current_max - current_min)) * new_max
    
    return normalized_arr

def _read_au_file(file_name: str, au: str) -> pd.DataFrame:
    df = pd.read_table(file_name, header=None, sep=None, engine="python", names=["index", au])
    df = df.drop("index", axis=1)
    return df

def _combine_au_files_subject(subject: str, aus: List[str], root_path: str) -> pd.DataFrame:
    list_of_au_dfs = []
    for au in aus:
        file_name = root_path / subject / f"{subject}_{au}.txt"
        df_au = _read_au_file(file_name, au)
        list_of_au_dfs.append(df_au)
    return pd.concat(list_of_au_dfs, axis=1)

def _get_au_dataframes(root_path: Path) -> Dict[str, pd.DataFrame]:
    subjects = [path.name for path in root_path.glob("S*")]
    subjects = sorted(subjects)
    aus = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    aus = ["au" + str(au) for au in aus]
    aus_subject = {}
    for subject in subjects:
        subject_df = _combine_au_files_subject(subject, aus, root_path)
        subject_df.insert(0, "subject", subject)
        aus_subject[subject] = subject_df
    return aus_subject


def _train_transforms(*args):
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform(*args)
    
def _preprocess_train(examples):
    images = [image.convert("RGB") for image in examples["image"]]
    examples["original_sizes"] = [image.size for image in images]
    examples["pixel_values"] = [_train_transforms(image) for image in images]
    examples["aus"] = [torch.tensor(sample) for sample in examples["aus"]]
    examples["caption"] = [caption for caption in examples["caption"]]
    return examples

def load_disfa(img_path: str, csv_path: str, caption_path: str) -> datasets.Dataset:
    csv_path = Path(csv_path)
    dict_of_dfs = _get_au_dataframes(csv_path)
    
    df = pd.concat(list(dict_of_dfs.values())).reset_index()
    # Get captions
    captions_df = pd.read_csv(caption_path)
    df["caption"] = captions_df["caption"]
    # Make AU columns be floats
    df = df.astype(dict(zip(df.loc[:, "au1": "au26"].columns.tolist(), 12 * ["float64"])))
    # Subsample only cases with atleast 1 active AU + add extra 8305 (10%) with no AUs
    df_sub = df[df.loc[:, "au1": "au26"].sum(1) > 0]
    df_au0 = df[df.loc[:, "au1": "au26"].sum(1) == 0].sample(n=8305)
    df = pd.concat([df_sub, df_au0]).reset_index()
    df = df.drop("level_0", axis=1)
    # Extract image data
    data_files = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = f'{img_path}/{row["subject"]}/{str(row["index"]).zfill(5)}.jpg'
        data_files.append(file_path)

    dataset_dict = {"image": data_files}
    disfa_dataset = datasets.Dataset.from_dict(dataset_dict).cast_column("image", RARImage())

    def mapping_disfa(x, idx):
        x["aus"] = df.loc[idx, "au1": "au26"]
        x["caption"] = df.loc[idx, "caption"]
        return x
    disfa_dataset = disfa_dataset.map(mapping_disfa, with_indices=True)
    disfa_dataset = disfa_dataset.with_transform(_preprocess_train)

    return disfa_dataset

def load_affectnet(rar_file: str, csv_path: str) -> datasets.Dataset:
    df = pd.read_csv(csv_path)
    # Filter images too small
    df = df[~df["HasFace"].isnull()]
    # Filter images that have isPhoto and IsReal
    df = df[df["IsPhoto"] & df["IsReal"]]
    # Filter images that are of size 512x512
    df = df[(df["ImageResolutionX"] >= 512) & (df["ImageResolutionY"] > 512)]
    # Filter low intensity images
    df = df[df.loc[:, "AU1_rec": "AU26_rec"].max(1) > 0.2]
    # Tone down AU1 and AU4 predictions, as they are too high
    mult_scale = 1.8
    df["AU1_rec"] = _normalize_to_range(df["AU1_rec"] ** mult_scale, df["AU1_rec"].max())
    df["AU4_rec"] = _normalize_to_range(df["AU4_rec"] ** mult_scale, df["AU4_rec"].max())
    # Increase AU 15, 17, 20 as they are too low
    mult_scale, amp_scale = 0.8, 1.8
    df["AU15_rec"] = _normalize_to_range(df["AU15_rec"] ** mult_scale, df["AU17_rec"].max()) * amp_scale
    df["AU17_rec"] = _normalize_to_range(df["AU17_rec"] ** mult_scale, df["AU17_rec"].max()) * amp_scale
    df["AU20_rec"] = _normalize_to_range(df["AU20_rec"] ** mult_scale, df["AU20_rec"].max()) * amp_scale
    # Multiply all AUs by 5 to reach intensity 0-5
    df.loc[:, "AU1_rec": "AU26_rec"] = 5 * df.loc[:, "AU1_rec": "AU26_rec"]
    # Clip all low values to 0s
    df_aus = df.loc[:, "AU1_rec": "AU26_rec"]
    df_aus[df_aus < 1.1] = 0
    df.loc[:, "AU1_rec": "AU26_rec"] = df_aus
    df = df.reset_index()

    #Extract image data
    with rarfile.RarFile(rar_file) as rf:
        pass
    dataset_dict = {"image": df["ImageName"].tolist()}
    an_dataset = datasets.Dataset.from_dict(dataset_dict).cast_column("image", RARImage(rf))
    
    def mapping_f(x, idx):
        x["aus"] = df.loc[idx, "AU1_rec": "AU26_rec"]
        x["caption"] = df.loc[idx, "TextDescription"]
        return x
    an_dataset = an_dataset.map(mapping_f, with_indices=True)
    an_dataset = an_dataset.with_transform(_preprocess_train)
    return an_dataset


if __name__ == "__main__":
    rar_file = "../datasets/affecnet/Manually_Annotated.part01.rar"
    csv_path = "../datasets/affecnet/metadata.csv"
    an_dataset = load_affectnet(rar_file, csv_path)
    print(an_dataset[2]["aus"])
    
    image_path = "../datasets/disfa/aligned"
    label_path = "../datasets/disfa/ActionUnit_Labels/"
    caption_path = "../datasets/disfa/disfa_captions.csv"
    disfa_dataset = load_disfa(image_path, label_path, caption_path)
    print(disfa_dataset[2]["aus"])






