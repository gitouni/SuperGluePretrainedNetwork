import argparse
import os
from functools import partial
import cv2
from tqdm import tqdm
from utils import refresh_dir
from marker import find_marker

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--input_root",type=str,default="slip")
    io_parser.add_argument("--input_subdir",type=str,default="markered")
    io_parser.add_argument("--output_subdir",type=str,default="marker")
    io_parser.add_argument("--save_fmt",type=str,default=".png")
    marker_parser = parser.add_argument_group()
    marker_parser.add_argument("--morphop_size",type=int,default=5)
    marker_parser.add_argument("--morphop_iter",type=int,default=1)
    marker_parser.add_argument("--morphclose_size",type=int,default=5)
    marker_parser.add_argument("--morphclose_iter",type=int,default=1)
    marker_parser.add_argument("--dilate_size",type=int,default=5)
    marker_parser.add_argument("--dilate_iter",type=int,default=2)
    marker_parser.add_argument("--marker_range",type=int,nargs=2,default=[145,255])
    marker_parser.add_argument("--value_threshold",type=int,default=90)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    subdirs = sorted(os.listdir(args.input_root))
    calib_find_marker = partial(find_marker,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morphop_size, args.morphop_size)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morphclose_size, args.morphclose_size)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate_size, args.dilate_size)),
        mask_range=args.marker_range,
        min_value=args.value_threshold,
        morphop_iter=args.morphop_iter,
        morphclose_iter=args.morphclose_iter,
        dilate_iter=args.dilate_iter
    )
    for subdir in subdirs:
        input_dir = os.path.join(args.input_root, subdir, args.input_subdir)
        output_dir = os.path.join(args.input_root, subdir, args.output_subdir)
        refresh_dir(output_dir)
        img_list = list(sorted(os.listdir(input_dir)))
        for img_name in tqdm(img_list, desc="segment markers in {}".format(input_dir)):
            img = cv2.imread(os.path.join(input_dir, img_name))
            marker_mask = calib_find_marker(img)
            cv2.imwrite(os.path.join(output_dir, os.path.splitext(img_name)[0]+args.save_fmt),marker_mask)