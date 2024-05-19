#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os


from models.matching import Matching
from models.utils import (make_matching_plot,
                          AverageTimer, read_image)
from tqdm import tqdm
import re

def extract_num(filename:str) -> int:
    return int(re.search(r'\d+',filename).group())

def options():
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_dir', type=str, default='dataset/colmap/00/images',
        help='Path to the directory that contains the images')
    parser.add_argument(
        "--n_src", type=int, default=3, help="sliding window size (half)"
    )
    parser.add_argument(
        '--output_dir', type=str, default='dataset/colmap/00/spg/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument("--save_descriptor",action='store_true',default=False)
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[-1],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    opt = parser.parse_args()
    print(opt)
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    return opt

if __name__ == '__main__':
    opt = options()
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    files = sorted(os.listdir(opt.input_dir))
    Nfile = len(files)
    pairs = []  # pairs of filenames
    for i in range(Nfile):
        jmin = max(0, i-opt.n_src)
        jmax = min(Nfile, i+opt.n_src)
        for j in range(jmin, jmax):
            if j == i:
                continue
            pairs.append([i,j])

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    kpt_output_dir = output_dir.joinpath('keypoint')
    kpt_output_dir.mkdir(exist_ok=True)
    match_output_dir = output_dir.joinpath('match')
    match_output_dir.mkdir(exist_ok=True)
    matching = Matching(config).eval().to(device)
    kpt_data = []
    num_list = []
    image_data = []
    with torch.inference_mode():
        for filename in tqdm(files, desc='extracting keypoints'):
            _, inp, _ = read_image(
                os.path.join(input_dir, filename), device, opt.resize, 0, opt.resize_float)
            kpt_dict:dict = matching.extract_keypoints(inp)
            kpt_data.append(kpt_dict)
            image_data.append(inp)
            num = extract_num(filename)
            num_list.append(num)
            save_data = {k:v[0].cpu().numpy() for k,v in kpt_dict.items()}
            if not opt.save_descriptor:
                save_data.pop('descriptors')
            np.savez(os.path.join(kpt_output_dir,"kpt_%04d.npz"%num), **save_data)
        t_iter = tqdm(pairs, desc='matching')
        for pair in t_iter:
            src_idx, tgt_idx = pair
            data = {k+'0':v for k,v in kpt_data[src_idx].items()}
            data.update({'image0':image_data[src_idx]})
            data.update({k+'1':v for k,v in kpt_data[tgt_idx].items()})
            data.update({'image1':image_data[tgt_idx]})
            pred = matching(data)
            matches = pred['matches0'][0].cpu().numpy()
            np.save(os.path.join(str(match_output_dir),"match_%04d_%04d.npy"%(num_list[src_idx],num_list[tgt_idx])), matches)
            t_iter.set_description("%d - %d"%(src_idx, tgt_idx))
            t_iter.update(1)
        t_iter.close()
        # how to extract correspondences:
        # valid = matches > -1
        # mkpts0 = kpts0[valid]
        # mkpts1 = kpts1[matches[valid]]

    