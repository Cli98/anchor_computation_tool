# Anchor computation tool

This repo primarily targets to help those who needs to compute anchors to customer dataset in object detection. Two type of tools have been implemented at this stage. They are,

1. An anchor visualization tool (anchor_inspector) to help you check if your anchor is suitable for your current dataset. If yes, then you do not need to modify your anchors. Else,
2. If no, then give a try to my implementation (k_mean_anchor_size) which is able to compute anchors for two-stage detectors and return anchor_scale + anchor_ratios. Those parameters are critical for object detection (for example in mmdetection). For some single-stage detector, you only need kmean results. Simple take them from my implementation.

The result has been tested on mmdetection framework with Faster Rcnn FPN algorithm, obtained an AP improvement of 2.2 points on typical aerial image detection dataset. This is a decent improvements from my perspective. 

# update log
[05-12-2020] As observed by jinfagang, passing boolean variable ("anchors" and "annotations") from terminal may not work. The author originally considers to load those parameters from yaml file only. An updated will be provided to allow passing boolean variables from terminal.

# Usage

For anchor_inspector, you need to provide a configuration file (.yml) together with the path to the dataset. If you are not able to access gui (maybe you host code on a server), then it is fine. Just enable --no_gui option. Else, you will be able to visualize your current input and annotations boxes on image. The green bbox indicates a match but red is not. So be alarmed if you see lots of red bboxs.

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

To use anchor_inspector, it is highly recommended to setup data in following structure:

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

If you need to modify your anchors, then try k_mean_anchor_size notebook! Run all ceils to get your anchor_scale and anchor_ratios. They will jump out to console at the end.

Good luck!

# Acknowledgement
I refer to and modify code from those two repos,

https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

https://github.com/zhouyuangan/K-Means-Anchors

Thanks to their high quality code. And please take your time to check their repo if you want.

# Development plan
I understand that it is necessary to provide more tutorial/examples on how anchor works. Maybe more visualization. To improve this repo,

1. I am currently conducting experiment on some open-source detection frameworks with an aerial dataset, as it contains more small objects. I will share some testing results once they are available. 

2. I'm also writing to wrap up what I know, as best as I can, for basic concepts and some tips you may need to develop your customer anchors. Stay tuned.

3. I'm not a computer vision phd and will always run into some errors/bugs. If you find some, please share your finding with everyone in this repo. You are welcome to PR.

4. If you fail to observe any improvements after you plug-in and play with this repo, please open an issue and describe how thing goes. I will take my best effort to help. 

# TODO list
The repo author is currently working on some side projects as well for the following two weeks. So there might be delays for those to-dos. Apologize for any inconvenience this may cause. And you're welcome to PR.

1. Build tutorial (shape dataset) and provide a working example.
2. Improve part of code to make it easier to use.
3. Revision for coord normalization for kmean code.
4. Pass boolean variable from terminal.
