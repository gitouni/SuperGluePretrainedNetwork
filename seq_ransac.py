import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RANSACRegressor
from utils.viz_utils import make_matching_plot
import cv2
import argparse
import os
from utils.utils import refresh_dir
import json
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--img_dir",type=str,default="slip/nail3/markered")
    io_parser.add_argument("--kpt_dir",type=str,default="res/nail3")
    io_parser.add_argument("--output_pos_dir",type=str,default="pos/nail3/gt")
    io_parser.add_argument("--output_fig_dir",type=str,default="debug_fig")
    io_parser.add_argument("--viz",type=bool,default=True)
    ransac_parser = parser.add_argument_group()
    ransac_parser.add_argument("--min_sample",type=int,default=5)
    ransac_parser.add_argument("--ransac_state",default=0)
    ransac_parser.add_argument("--residual_threshold",type=float,default=5)
    return parser.parse_args()

def rmse(x:np.ndarray, y:np.ndarray):
    return np.sqrt(np.sum((x-y)**2, axis=1).mean())


def get_xyth(x1:np.ndarray, x2:np.ndarray):
    x1_mean:np.ndarray = np.mean(x1, axis=0, keepdims=True)
    x2_mean:np.ndarray = np.mean(x2, axis=0, keepdims=True)
    x1_:np.ndarray= x1 - x1_mean
    x2_:np.ndarray = x2 - x2_mean
    H = np.dot(x1_.T, x2_)  # (2,2)
    U, _ ,V = np.linalg.svd(H)
    R = np.dot(V, U.T)
    if np.linalg.det(R) < 0:
        R[-1,:] = -R[-1,:]
    t = x2_mean.squeeze() - R @ x1_mean.squeeze()
    return R, t

def transform2d(R:np.ndarray, t:np.ndarray, x:np.ndarray):
    return np.transpose(R @ x.T + t.reshape(2,1))

class SolveEstimator(BaseEstimator,RegressorMixin):
    def __init__(self,rot=np.eye(2), tsl=np.zeros(2)) -> None:
        self.rot = rot
        self.tsl = tsl
    
    def fit(self,x:np.ndarray,y:np.ndarray):
        R, t = get_xyth(x, y)
        self.rot = R
        self.tsl = t
        return self
    
    def score(self,x:np.ndarray,y:np.ndarray):
        return rmse(transform2d(self.rot, self.tsl, x), y)
    
    def predict(self,x:np.ndarray):
        return transform2d(self.rot, self.tsl, x)
    
    def get_params(self, deep=True):
        return super().get_params(deep)
    
    def set_params(self, **params):
        return super().set_params(**params)

def step_ransac(args, kpt0:np.ndarray, kpt1:np.ndarray):
    base_estimator = SolveEstimator()
    ransac_estimator = RANSACRegressor(estimator=base_estimator, min_samples=args.min_sample, random_state=args.ransac_state, residual_threshold=args.residual_threshold)
    ransac_estimator.fit(kpt0, kpt1)
    param = ransac_estimator.estimator_.get_params()
    valid = ransac_estimator.inlier_mask_
    R = param['rot']
    t = param['tsl']
    return R, t, kpt0[valid], kpt1[valid]

if __name__ == "__main__":
    args = options()
    kpt_files = sorted(os.listdir(args.kpt_dir))
    refresh_dir(args.output_pos_dir)
    refresh_dir(args.output_fig_dir)
    for kpt_file in tqdm(kpt_files):
        kpt_data = json.load(open(os.path.join(args.kpt_dir, kpt_file)))
        src_kpt = np.array(kpt_data['src_kpt'], dtype=np.float32)
        tgt_kpt = np.array(kpt_data['tgt_kpt'], dtype=np.float32)
        R, t, kpt0, kpt1 = step_ransac(args, src_kpt, tgt_kpt)
        src_file = kpt_data['src_file']
        tgt_file = kpt_data['tgt_file']
        src_core = os.path.splitext(src_file)[0]
        tgt_core = os.path.splitext(tgt_file)[0]
        with open(os.path.join(args.output_pos_dir,"matching_{}_{}.txt".format(src_core, tgt_core)), 'w')as f:
            f.write("{:0.4f} {:0.4f} {:0.4f}".format(np.arccos(R[0,0]), t[0], t[1]))
        if args.viz:
            img1 = cv2.imread(os.path.join(args.img_dir, src_file))
            img2 = cv2.imread(os.path.join(args.img_dir, tgt_file))
            matched_fig = make_matching_plot(img1, img2, kpt0, kpt1)
            cv2.imwrite(os.path.join(args.output_fig_dir,"matching_{}_{}.png".format(src_core, tgt_core)), matched_fig)