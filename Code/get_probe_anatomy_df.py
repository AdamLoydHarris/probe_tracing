"""
This library contains preprocessing functions that load preprocessed histology data that has been aligned
to the Allen mouse brain atlas (CCF coordinates) with HERBS (https://github.com/JingyiGF/HERBS), and uses the AllenSDK
API to convert the voxel coordinates of each site on the Neuropixel probe to anatomical information 
(structure ID, acronym, name, rgb_triplet).

Note: this code must be run in an environment with allenSDK installed and an internet connection to download the Allen
mouse brain atlas data.

adapted from code by @peterdoohan
"""

# %% Imports
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import shutil
import ast
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# %% Global variables

ALLEN_ATLAS_RESOLUTION = 10  # um (10, 25 or 50)
HERBS_ATLAS_SHAPE = (1140, 1320, 800)  # (z,x,y) voxels

DATA_PATH = Path("../data")  # path to top level data folder

HERBS_DATA_FOLDER = "/Users/AdamHarris/Documents/probe_tracing/HERBS_outputs" #DATA_PATH / "preprocessed_data" / "HERBS"  # folder containing subject_ID/proble.pkl files

# Load Allen anatomy objects from web
MCC = MouseConnectivityCache(resolution=ALLEN_ATLAS_RESOLUTION)
REFERENCE_SPACE = MCC.get_reference_space()

NEUROPIXEL_RECORDING_CHANNELS = np.arange(
    0, 384, dtype=int
)  # list or array of channel IDs recorded on the Neuropixel probe

CNT_O_RECORDING_CHANNELS = np.arange(
    0, 10, dtype=int
)

CNT_I_RECORDING_CHANNELS = np.arange(
    0, 11, dtype=int
)

NEUROPIXELS_MICE = ["ab03", "ah07"]

PROCESSED_DATA_PATH = DATA_PATH / "processed_data"
# %% Functions


def get_probe_anatomy_df(subject_ID):
    """ """
    HERBS_probe_path = f"{HERBS_DATA_FOLDER}/probe {subject_ID}.pkl" #probe.pkl"
    if subject_ID in NEUROPIXELS_MICE:
        NUM_CHANNELS = NEUROPIXEL_RECORDING_CHANNELS
    elif subject_ID[-1] in ["A", "F"]:
        NUM_CHANNELS = CNT_I_RECORDING_CHANNELS
    else:
        NUM_CHANNELS = CNT_O_RECORDING_CHANNELS
    
    print(NUM_CHANNELS)
    voxel_coords = get_probe_tract_as_numpy(HERBS_probe_path)
    structure_ids = REFERENCE_SPACE.annotation[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
    structure_tree = MCC.get_structure_tree()
    structures = structure_tree.get_structures_by_id(structure_ids.tolist())  # None = outside the brain
    probe_site_anatomy = []
    for i, (voxel, structure) in enumerate(zip(voxel_coords, structures)):
        if structure is None:
            probe_site_anatomy.append(
                {
                    "channel_no": i + 1,
                    "voxel": tuple(voxel),
                    "structure_ID": 0,
                    "acronym": "",
                    "name": "Outside brain",
                    "rgb_triplet": (128, 128, 128),  # grey
                }
            )
        else:
            probe_site_anatomy.append(
                {
                    "channel_no": i + 1,
                    "voxel": tuple(voxel),
                    "structure_ID": structure["id"],
                    "acronym": structure["acronym"],
                    "name": structure["name"],
                    "rgb_triplet": tuple(structure["rgb_triplet"]),
                }
            )
    probe_anatomy_df = pd.DataFrame(probe_site_anatomy)
    if len(probe_anatomy_df) > len(NUM_CHANNELS):
        # exclude channles not recorded from
        probe_anatomy_df = probe_anatomy_df.loc[NUM_CHANNELS]
    elif len(probe_anatomy_df) < len(NUM_CHANNELS):
        # assume missing channels were outside brain and not regestered in HERBS
        n_missing_channels = len(NUM_CHANNELS) - len(probe_anatomy_df)
        missing_channels = pd.DataFrame(
            [
                {
                    "channel_no": channel_ID + 1,
                    "voxel": np.nan,
                    "structure_ID": 0,
                    "acronym": "",
                    "name": "Outside brain",
                    "rgb_triplet": (128, 128, 128),
                }
                for channel_ID in range(len(probe_anatomy_df), len(probe_anatomy_df) + n_missing_channels)
            ]
        )
        probe_anatomy_df = pd.concat([probe_anatomy_df, missing_channels], ignore_index=True)
    return probe_anatomy_df


def get_probe_tract_as_numpy(HERBS_probe_path):
    """
    Load probe tract from HERBS output probe.pkl and return as numpy array
    of shape (n_probe_sites, 3[x,y,z])
    """
    with open(HERBS_probe_path, "rb") as file:
        HERBS_data = pickle.load(file)
    probe_coords = HERBS_data["data"]["sites_vox"][0]  # voxel coordinates (z,y,x) origin bottom left of left cerebellum
    z,x,y = probe_coords.T
    # translate to origin top right of right olfactory bulb (Allen standard)
    z = HERBS_ATLAS_SHAPE[0] - z
    y = HERBS_ATLAS_SHAPE[2] - y
    x = HERBS_ATLAS_SHAPE[1] - x
    # remap origin
    probe_coords = np.array([x, y, z]).T
    # for some reson HERBS output only lists a voxel for every second site, assume adjacent sites lie in the same voxel
    probe_coords = np.repeat(probe_coords, repeats=2, axis=0)
    return probe_coords.astype(int)


def remove_allen_cached_data():
    """Allen will automatically download and save data directly to the working directory.
    Remove this data after processing"""
    for folder in ["mouse_connectivity"]:
        shutil.rmtree(folder)
    return print("Allen cached data removed")


if __name__ == '__main__':
    subject_ID = 'ah03_F'
    df = get_probe_anatomy_df(subject_ID)
    df.to_csv(f"{HERBS_DATA_FOLDER}/{subject_ID}.csv")
    
# %%
# %%

# %%
remove_allen_cached_data()
