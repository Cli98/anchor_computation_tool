#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import cv2
import datetime
import os
import argparse
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

# Set keycodes for changing images
# 81, 83 are left and right arrows on linux in Ascii code (probably not needed)
# 65361, 65363 are left and right arrows in linux
# 2424832, 2555904 are left and right arrows on Windows
# 110, 109 are 'n' and 'm' on mac, windows, linux

"""
Particularly helpful is the --annotations flag which displays your annotations on the images from your dataset. 
Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. 
If an annotation doesn't have anchors available, it means it won't contribute to training. 
It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause 
for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

Code adapted and modified from keras-retinanet repo and served as supplement code for zylo117's implementation.
"""

from anchor_utils import draw_annotations, draw_boxes, draw_caption, compute_gt_annotations
from efficientdet.dataset import CocoDataset, Resizer, collater

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

class Anchors(nn.Module):
    """
    Adapted and modified from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/utils.py
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.
        How to compute total number of possible anchors: keras-retinanet issue 60
        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        print("Loading image with numpy array")
        image_shape = image.shape[:2]
        device = "cpu"
        if image_shape == self.last_shape and device in self.last_anchors:
            return self.last_anchors[device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape
        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale

                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[device] = anchor_boxes
        return anchor_boxes


def parse_args(args):
    """ Parse the arguments.
    """
    #TODO: Check min/max resize parameter suitable
    #TODO: Extract anchor parameters from config file. Require ratio, scale, size, stride
    parser     = argparse.ArgumentParser(description='Debug script for proper anchors.')
    parser.add_argument('-project_name', type=str, help='The path for project')
    parser.add_argument('-dataset_path', type=str, help='The path for source data')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', default= False, help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', default = True, help='Show annotations on the image. Green annotations '
                                              'have anchors, red annotations don\'t and '
                                              'therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')
    parser.add_argument('--no-gui', default = True, help='Do not open a GUI window. Save images to an output directory instead.', action='store_true')
    parser.add_argument('--limit', default=10,
                        help='Only available in no-gui option. Draw given number of image',
                        action='store_true')
    parser.add_argument('--output-dir', help='The output directory to save images to if --no-gui is specified.', default='.')
    parser.add_argument('--flatten-output', help='Flatten the folder structure of saved output images into a single folder.', action='store_true')

    return parser.parse_args(args)


def run(generator, args, anchor_params, draw_paras):
    """ Main loop.

    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    for i, data in enumerate(generator):
        # load the data
        image, annotations = np.transpose(data['img'].numpy(), (0, 2, 3, 1))[0], data['annot'][0].numpy()
        if len(annotations) > 0:
            anchors = anchor_params(image).numpy()[0]
            #print("anchors: ", anchors[-5:,:])
            # print("annotations: ",annotations[:5,:4])
            #print("scale and ratio: ", anchor_params.scales, anchor_params.ratios)
            positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations[:,:4]) # best 1 anchor
            # draw anchors on the image
            # print("parameters: ",args.anchors,args.annotations,args.display_name)
            if draw_paras['no_gui']:
                image = image*255
                image = image.astype("uint8")

            if draw_paras['anchors']:
                #print("length of anchors: ",len(positive_indices),anchors.shape,image.shape)
                #print(len(anchors[positive_indices]))
                image = draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=3)

            # draw annotations on the image
            if draw_paras['annotations']:
                # draw annotations in red
                draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=None)

                # draw regressed anchors in green to override most red annotations
                # result is that annotations without anchors are red, with anchors are green
                draw_boxes(image, annotations[:,:4][max_indices[positive_indices], :], (0, 255, 0))

        # write to file and advance if no-gui selected
        if draw_paras['no_gui']:
            output_path = make_output_path(args.output_dir, str(i)+".jpg", flatten=args.flatten_output)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            if args.limit and i+1>=args.limit:
                return
            continue
        # if we are using the GUI, then show an image
        cv2.imshow('Image', image)
        print("Image id: ",i)
        key = cv2.waitKeyEx()
        if (key == ord('q')) or (key == 27):
            cv2.destroyAllWindows()
            return False
    return True


def make_output_path(output_dir, image_path, flatten = False):
    """ Compute the output path for a debug image. """

    # If the output hierarchy is flattened to a single folder, throw away all leading folders.
    if flatten:
        path = os.path.basename(image_path)

    # Otherwise, make sure absolute paths are taken relative to the filesystem root.
    else:
        # Make sure to drop drive letters on Windows, otherwise relpath wil fail.
        _, path = os.path.splitdrive(image_path)
        if os.path.isabs(path):
            path = os.path.relpath(path, '/')

    # In all cases, append "_debug" to the filename, before the extension.
    base, extension = os.path.splitext(path)
    path = base + "_debug" + extension

    # Finally, join the whole thing to the output directory.
    return os.path.join(output_dir, path)


def create_generator(args, params, input_size):
    # Visualize anchors to see if they match for per image
    # For visualization purpose, no argumentation will be applied at here, else you may see distortion.
    # Except resize op, as you will resize your image when you train it anyway.
    training_params = {'batch_size': 1,
                       'shuffle': False,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': args.num_workers}

    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': args.num_workers}
    training_set = CocoDataset(root_dir=args.dataset_path, set=params.train_set,
                               transform=transforms.Compose([Resizer(input_size)]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=args.dataset_path, set=params.val_set,
                          transform=transforms.Compose([Resizer(input_size)]))
    val_generator = DataLoader(val_set, **val_params)

    return training_generator, val_generator

def test_generator(generator):
    # DEBUG purpose only, check if generator achieves desired functionality.
    # DO NOT USE IF YOU HAVE NO IDEA HOW IT WORKS
    # Testing objective: properly load images and annotations
    # On May 3rd, test passed.
    class_list = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle",
                  "bus", "motor"]
    class_list = {idx: val for idx, val in enumerate(class_list)}
    data = next(iter(generator))
    image, anno = np.transpose(data['img'].numpy(),(0,2,3,1))[0], data['annot'][0].numpy()
    for i in range(anno.shape[0]):
        x1, y1, x2, y2, class_id = anno[i].astype(np.int16).tolist()
        # print(x1, y1, x2, y2, class_id)
        # coord format: x1,y1,x2,y2
        image = cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
        text = class_list[class_id]
        cv2.putText(image, text, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    lineType=cv2.LINE_AA)
    cv2.imshow("Test img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

"""
Use yaml file to pass anchor visualization parameters
"""
def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    params = Params(f'{args.project_name}.yml')
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
    compoend_coef = 4
    # create the generator
    training_generator, val_generator = create_generator(args, params, input_size=input_sizes[compoend_coef])
    # test_generator(training_generator)
    # pass a dict should solve the issues
    anchor_dict = {'scales':eval(params.anchors_scales), "ratios":eval(params.anchors_ratios)}
    draw_para = {'anchors':params.anchors, 'annotations':params.annotations, 'no_gui':params.no_gui}
    anchor_params = Anchors(anchor_scale=anchor_scale[compoend_coef], **anchor_dict)

    # create the display window if necessary
    if not args.no_gui:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    run(training_generator, args, anchor_params=anchor_params, draw_paras = draw_para)


if __name__ == '__main__':
    main()
