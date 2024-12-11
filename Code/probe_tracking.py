"""
This library contains functions for visualising the anatomical location of the Neuropixel and 
Cambridge Neurotech F-Series probes.
Here we load processed histology data probe.csv from each subject.
See preprocessing module for how this data is generated.

Note: this code must be run in an environment with brainrender installed 
a YAML to create the conda environmennt with requisite


Core functions adapted from code by @peterdoohan
"""

# %% Imports
import ast
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from brainrender import Scene
from brainrender.actors import Line
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


import brainrender

brainrender.settings.SHADER_STYLE = "plastic"

import vedo

vedo.settings.default_backend = "vtk"
# %% Global variables
ALLEN_ATLAS_RESOLUTION = 10  # um (10, 25, 50)
# PROCESSED_DATA_PATH = Path("../data/processed_data")

# EXPERIMENT_INFO_PATH = Path("../data/experiment_info")
# with open(EXPERIMENT_INFO_PATH / "subject_IDs.json", "r") as infile:
    # SUBJECT_IDS = json.load(infile)

SUBJECT_IDS = ["ab03", "ah07", 
               "ah03_A", "ah03_B", "ah03_C", "ah03_D", "ah03_E", "ah03_F",
               "ah04_A", "ah04_B", "ah04_C", "ah04_D", "ah04_E", "ah04_F",
               "me08_A", "me08_B", "me08_C", "me08_D", "me08_E", "me08_F",
               "me10_A", "me10_B", "me10_C", "me10_D", "me10_E", "me10_F",
               "me11_A", "me11_B", "me11_C", "me11_D", "me11_E", "me11_F"]

HERBS_DATA_PATH = "/Users/AdamHarris/Documents/probe_tracing/HERBS_outputs"

NEUROPIXELS_MICE = ["ab03", "ah07"]

EXTRAPOLATE_CNT = False

ah07, ab03, ah03, ah04, me08, me10, me11 = sns.color_palette('tab10', 7) 
MOUSE_COLOURS_DICT = {
    'ah03': ah03,
    'ah04': ah04,
    'ah07': ah07,
    'ab03': ab03,
    'me08': me08,
    'me10': me10,
    'me11': me11
}

# %% Main functions

def load_probe_anatomy_df(subject_ID):
    df = pd.read_csv(f"{HERBS_DATA_PATH}/{subject_ID}.csv")
    return df

def smooth_coordinates(coordinates, window_size=3):
    """
    Smooth the 3D coordinates using a moving average.

    :param coordinates: List of tuples, where each tuple represents a 3D coordinate (x, y, z).
    :param window_size: Integer specifying the size of the moving average window.
    :return: List of tuples representing the smoothed coordinates.
    """
    smoothed_coordinates = []
    coords_array = np.array(coordinates)
    for i in range(len(coordinates)):
        start = max(0, i - window_size // 2)
        end = min(len(coordinates), i + window_size // 2 + 1)
        window = coords_array[start:end]
        smoothed_coord = np.mean(window, axis=0)
        smoothed_coordinates.append(tuple(smoothed_coord))
    return smoothed_coordinates

def extrapolate_along_line(coordinates, num_bins, direction='forward'):
    """
    Extrapolate a list of 3D coordinates along the line formed by the coordinates in a specified direction.

    :param coordinates: List of tuples, where each tuple represents a 3D coordinate (x, y, z).
    :param num_bins: Integer specifying the number of bins to extrapolate.
    :param direction: String specifying the direction to extrapolate ('forward' or 'backward').
    :return: List of tuples representing the extrapolated coordinates.
    """
    if len(coordinates) < 2:
        raise ValueError("At least two coordinates are required to determine the line direction.")

    # Compute the direction vector from the first two coordinates
    p1, p2 = coordinates[0], coordinates[1]
    direction_vector = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    
    # Calculate the magnitude of the direction vector
    magnitude = (direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2) ** 0.5
    
    if magnitude == 0:
        raise ValueError("The first two coordinates are identical, resulting in a zero direction vector.")
    
    # Normalize the direction vector
    unit_direction = (direction_vector[0] / magnitude, direction_vector[1] / magnitude, direction_vector[2] / magnitude)
    
    # Determine the starting point for extrapolation
    if direction == 'forward':
        start_coord = coordinates[-1]
    elif direction == 'backward':
        start_coord = coordinates[0]
        unit_direction = (-unit_direction[0], -unit_direction[1], -unit_direction[2])
    else:
        raise ValueError("Invalid direction. Choose 'forward' or 'backward'.")

    # Extrapolate the coordinates
    extrapolated_coordinates = []
    for i in range(1, num_bins + 1):
        new_coord = (
            start_coord[0] + i * unit_direction[0],
            start_coord[1] + i * unit_direction[1],
            start_coord[2] + i * unit_direction[2]
        )
        extrapolated_coordinates.append(new_coord)

    coordinates.extend(extrapolated_coordinates)

    return coordinates

def plot_experiment_probe_tracts(full_brain=True, visualise_regions=["PL1"]):
    """
    Plots probe tract from all subjects into brainrender scene
    Args:
    - full_brain: bool, if True, plot full brain, if False, plot only visualise_regions
    - visualise_regions: list of strings, list of brain regions to visualise
    """
    # initialise brainrender scene
    scene = Scene(title="", root=full_brain)
    scene.plotter.axes = False
    for region in visualise_regions:
        if "IL" in region:
            c = 'cyan'
        elif "MO" in region:
            c = 'green'
        elif "AC" in region:
            c = 'yellow'

        else:
            c = 'magenta'
            
        scene.add_brain_region(region, alpha=0.10, hemisphere="left", color=c, silhouette=False)
    # add probe tracts
    print(f'subject_ids = {SUBJECT_IDS}')
    subject_palette = sns.color_palette("Set2", len(SUBJECT_IDS)).as_hex()
    for subject, color in zip(SUBJECT_IDS, subject_palette):
        print(f"subject = {subject}")
        probe_anatomy_df = load_probe_anatomy_df(subject)
        # load (n_sites in the brain, 3) np.array of voxel coordinates
        probe_track = np.array(probe_anatomy_df.voxel.dropna().to_list())

        probe_track = [ast.literal_eval(loc) for loc in probe_track]
        # probe_track_um = probe_track * ALLEN_ATLAS_RESOLUTION  # convert to um
        probe_track_um = [(ALLEN_ATLAS_RESOLUTION * e for e in tup) for tup in probe_track]
        probe_track_um = [tuple(tup) for tup in probe_track_um]
        if EXTRAPOLATE_CNT:
            if subject not in NEUROPIXELS_MICE:
                probe_track_um = smooth_coordinates(probe_track_um, window_size=7)
                probe_track_um = extrapolate_along_line(probe_track_um, num_bins=1000)
        c = MOUSE_COLOURS_DICT[subject[:4]]


        scene.add(Line(probe_track_um, color=c, linewidth=6))
    scene.render(interactive=True)
    scene.screenshot("/Users/AdamHarris/Desktop/probes_in_brain.svg")
    return scene


def plot_probes_with_anatomical_locations(subject_IDs="all", custom_colors=True):
    """
    Visualises anatomical locations along Neuropixel probes for select subjects.
    Args:
    - subject_IDs: list of strings, list of subject IDs to visualise, or "all" to visualise all subjects
    - custom_colors: bool, if True, use custom colors for brain regions, if False, use AllenSDK colors
    """ #SUBJECT_IDS if subject_IDs == "all" else subject_IDs
 
    f, axes = plt.subplots(1, len(subject_IDs), figsize=(len(subject_IDs) / 2, 7))
    f.subplots_adjust(wspace=0.5)
    region2custom_color = _get_custom_region_colors() #, non_colours="Reds")
    regions_acr = _get_all_regions_recorded()
    regions_fullnames = _get_all_regions_recorded_fullname()
    print(regions_fullnames)
    print(regions_acr)
    regions_dic = {}
    for acr, name in zip(regions_acr, regions_fullnames):
        regions_dic[acr] = name
    print(regions_dic)
    for subject, ax in zip(subject_IDs, axes):
        print(subject)
        probe_df = load_probe_anatomy_df(subject)
        if custom_colors:
            probe_df["custom_rgb"] = probe_df.acronym.map(region2custom_color)

            site_colors = np.array(probe_df.custom_rgb.to_list())
        else:
            site_colors = np.array(probe_df.rgb_triplet.to_list())

        if subject in NEUROPIXELS_MICE:
            site_colors = np.array(site_colors).reshape((192, 2, 3))  # approx probe geom as 2 by 192 sites
        else:
            site_colors = np.expand_dims(site_colors, axis=1) # treat cambridge neurotech as 1d (only 10 or 11 sites per shank)

        ax.imshow(site_colors, aspect="auto")
        ax.axis("off")
        ax.invert_yaxis()
        ax.set_title(subject, rotation=45)
    # change nan to "Outside brain" and add legend
    regions = list(region2custom_color.keys())
    print(regions)
    reg_full = [regions_dic[ac] for ac in regions]
    print(reg_full)
    regions[0] = "Outside brain"
    colors = list(region2custom_color.values())
    legend_elements = [Patch(facecolor=np.array(color), label=region) for region, color in zip(reg_full, colors)]
    f.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1, fontsize=10, frameon=False)
    f.savefig('/Users/AdamHarris/Documents/probe_tracing/channel_regions.pdf',  bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return



# %% Supporting Functions


def _get_custom_region_colors():
    # mEC colors
    PL1, PL2_3, PL5, PL6a = sns.color_palette("RdPu", 4)
    ILA2_3, ILA5, ORBm5, ORBm6a = sns.color_palette("Blues", 4)
    MOs1, MOs2_3, MOs5 = sns.color_palette("Greens", 3)
    ACAv5, ACAv6a, ACAd5 = sns.color_palette("Wistia", 3)
    OLF, STR, TTd, ccg, DP, LSr, _, _ = sns.color_palette("Greys", 8)

    
    # ENTm1, ENTm2, ENTm3, ENTm5, ENTm6, ENTl5 = sns.color_palette(ENT_colors, 6)
    # VISpl1, VISp1, VISl1, VISpl2_3, VISp2_3, VISpl4, VISp4, VISpl5, VISpor5, VISpl6a = sns.color_palette(VIS_colors, 10)
    # VIS colors
    region2color = {
        np.nan: (0, 0, 0),
        "MOs1": MOs1,
        "MOs2/3": MOs2_3, 
        "MOs5": MOs5,
        "ACAv5": ACAv5,
        'ACAv6a': ACAv6a,
        "ACAd5": ACAd5, 
        "PL1": PL1,
        "PL2/3": PL2_3,
        "PL5": PL5,
        'PL6a': PL6a,
        'ILA2/3': ILA2_3,
        "ILA5": ILA5,
        "ORBm5":ORBm5,
        'ORBm6a':ORBm6a,
        "OLF": OLF,
        "STR": STR,
        "TTd": TTd,
        "ccg": ccg,
        "DP": DP,
        "LSr": LSr,
    }


    # Check all regions have a color
    all_regions = _get_all_regions_recorded()
    all_regions_flag = all([region in region2color for region in all_regions])
    missing_regions = [region for region in all_regions if region not in region2color]
    assert all_regions_flag, f"Missing region color for {missing_regions}"
    return region2color


def _get_all_regions_recorded():
    """ """
    regions = []
    for subject in SUBJECT_IDS:
        probe_df = load_probe_anatomy_df(subject)
        regions.append(probe_df.acronym)
    return pd.concat(regions).unique()

def _get_all_regions_recorded_fullname():
    """ """
    regions = []
    for subject in SUBJECT_IDS:
        probe_df = load_probe_anatomy_df(subject)
        regions.append(probe_df.name)
    return pd.concat(regions).unique()

# %%
if __name__ == "__main__":
    plot_probes_with_anatomical_locations(subject_IDs=SUBJECT_IDS, custom_colors=True)
    # plot_experiment_probe_tracts(full_brain=True, visualise_regions=['ILA5', 
    #                                                                  'PL5', 
    #                                                                  'PL6a',
    #                                                                  "ACAv2/3", 
    #                                                                  'ACAd5', 
    #                                                                  'ACAd2/3', 
    #                                                                  "ACAv5", 
    #                                                                  "ACAd1",
    #                                                                  'MOs2/3', 
    #                                                                  'MOs1'])
    
    # plot_probes_with_anatomical_locations(subject_IDs= ["ab03"], custom_colors=True)
# %%
# regions = _get_all_regions_recorded()
# print(regions)
# PL5 ,PL2_3, PL1, PL6a = sns.color_palette("RdPu", 4)
# ILA5,ORBm5, ORBm6a, ILA2_3 = sns.color_palette("Blues", 4)
# MOs2_3, MOs1, MOs5 = sns.color_palette("Greens", 3)
# ACAv2_3, ACAd5, ACAd2_3, ACAv2_3, ACAd1, ACAv5, ACAv6a, ACAd6a = sns.color_palette("Wistia", 8)
# STR,SH, ccg, OLF, TTd, DP, LSr, _, _ = sns.color_palette("Greys", 9)

# region2color = {
#         np.nan: (0, 0, 0),
        
#         "MOs2/3": MOs2_3, 
#         "MOs1": MOs1,
#         "MOs5": MOs5,
#         "ACAd1": ACAd1,
#         "ACAv2/3": ACAv2_3, 
#         "ACAd5": ACAd5, 
#         "ACAd2/3": ACAd2_3, 
#         "ACAv2/3": ACAv2_3,
#         "ACAv5": ACAv5,
#         'ACAv6a': ACAv6a,
#         'ACAd6a':ACAd6a,
#         'PL6a': PL6a,
#         "PL5": PL5,
#         "PL2/3": PL2_3,
#         "PL1": PL1,
#         "ILA5": ILA5,
#         'ILA2/3': ILA2_3,
#         "ORBm5":ORBm5,
#         'ORBm6a':ORBm6a,
#         "OLF": OLF,
#         "TTd": TTd,
#         "DP": DP,
#         "STR": STR,
#         "SH": SH, 
#         "ccg": ccg,
#         "LSr": LSr,
#         "PAR": (1.0, 1.0, 0.0),
#     }

# # %%

# # %%

# %%

MOUSE_COLOURS_DICT = {
    'ah03': ah03,
    'ah04': ah04,
    'ah07': ah07,
    'ab03': ab03,
    'me08': me08,
    'me10': me10,
    'me11': me11
}
fig, ax = plt.subplots(figsize=(3, len(MOUSE_COLOURS_DICT) * 0.3))

# Hide the axes
ax.axis('off')

# Add a legend with colored squares
for i, (name, color) in enumerate(MOUSE_COLOURS_DICT.items()):
    ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
    ax.text(1.5, i + 0.5, name, va='center', fontsize=12)

# Adjust the figure and display
plt.xlim(0, 4)
plt.ylim(0, len(MOUSE_COLOURS_DICT))
plt.gca().invert_yaxis()  # Invert y-axis to have first item on top
plt.show()
# %%

fig, ax = plt.subplots(figsize=(2, len(MOUSE_COLOURS_DICT) * 0.6))

# Hide the axes
ax.axis('off')

# Define the amount of whitespace between patches
spacing = 0.5

# Add a legend with colored squares
for i, (name, color) in enumerate(MOUSE_COLOURS_DICT.items()):
    y_position = i * (1 + spacing)
    ax.add_patch(plt.Rectangle((0, y_position), 1, 1, color=color))
    ax.text(1.5, y_position + 0.5, name, va='center', fontsize=12)

# Adjust the figure and display
plt.xlim(0, 4)
plt.ylim(0, len(MOUSE_COLOURS_DICT) * (1 + spacing))
plt.gca().invert_yaxis()  # Invert y-axis to have first item on top
plt.show()
fig.savefig('/Users/AdamHarris/Documents/probe_tracing/probe_legend.pdf',  bbox_inches='tight', pad_inches=0.1)


# %%
