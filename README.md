# Anchor computation tool

This repo primarily target to help those who need to compute their anchors to customer dataset. Two type of tools have been implemented at this stage. They are,

1. An anchor visualization tool (anchor_inspector) to help you check if your anchor is suitable for your current dataset. If yes, then you do not need to replace your anchors. 
2. If no, then give a try to my implementation (k_mean_anchor_size) which is able to compute anchors for two-stage detectors and return anchor_scale + anchor_ratios.

# Usage

For anchor_inspector, you need to provide a configuration file (.yml) together with path of dataset. If you are not able to access gui, then it is fine. Just enable --no_gui option. Else, you will be able to visualize your current input and annotations boxes on image. The green bbox indicates a match but red is not. So be alarmed if you see lots of red bboxs.

Some examples:

![alt text](https://github.com/Cli98/anchor_computation_tool/blob/master/images/anchor_example.png "Some visualized anchors")

Usage: anchor_inspector.py [-h] [-project_name PROJECT_NAME]
                           [-dataset_path DATASET_PATH] [-n NUM_WORKERS]
                           [--no-resize] [--anchors] [--annotations]
                           [--random-transform]
                           [--image-min-side IMAGE_MIN_SIDE]
                           [--image-max-side IMAGE_MAX_SIDE] [--config CONFIG]
                           [--no-gui] [--output-dir OUTPUT_DIR]
                           [--flatten-output]

Recommended to setup data in following structure:

Root

----------Datasets

--------------Train data

--------------Valid data!

--------------Test data

----------Projects

--------------project.yaml

Simple code to run:
python anchor_inspector.py -dataset_path ./datasets/a -project_name ./projects/a_project --output-dir debug_out

# How to get customer anchor

If you need to modify your anchors, then play with k_mean_anchor_size notebook. Run all notebook to get your anchor_scale and anchor_ratios. They will jump out to console at end.

Good luck!

