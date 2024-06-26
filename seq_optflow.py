import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RANSACRegressor
from models.utils import make_matching_plot
import cv2
import argparse
import os
from utils.utils import refresh_dir
from tqdm import tqdm


def options():
    parser = argparse.ArgumentParser()
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--img_dir",type=str,default="data/slip/knife1/NS")
    io_parser.add_argument("--output_pos_dir",type=str,default="res/slip/knife1/NS")
    io_parser.add_argument("--output_fig_dir",type=str,default="res/slip/knife1/fig")
    io_parser.add_argument("--viz",type=bool,default=False)
    feature_parser = parser.add_argument_group()
    feature_parser.add_argument("--maxCorners",type=int, default=200)
    feature_parser.add_argument("--qualityLevel",type=float,default=0.05)
    feature_parser.add_argument("--minDistance",type=int,default=7)
    feature_parser.add_argument("--blockSize",type=int,default=7)
    lk_parser = parser.add_argument_group()
    lk_parser.add_argument("--winSize",type=int, nargs=2, default=[15,15])
    lk_parser.add_argument("--maxLevel",type=int, default=3)
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

def step_optical(args, img1:np.ndarray, img2:np.ndarray):
    if np.ndim(img1) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1  # reference copy but read-only
    if np.ndim(img2) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2  # reference copy but read-only
    p0 = cv2.goodFeaturesToTrack(gray1, mask = None, maxCorners=args.maxCorners, qualityLevel=args.qualityLevel, minDistance=args.minDistance, blockSize=args.blockSize)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, winSize=args.winSize, maxLevel=args.maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    kpt0 = p0[st == 1]
    kpt1 = p1[st == 1]
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
    img_files = sorted(os.listdir(args.img_dir))
    refresh_dir(args.output_pos_dir)
    if args.viz:
        refresh_dir(args.output_fig_dir)
    for name1, name2 in tqdm(zip(img_files[:-1], img_files[1:]),total=len(img_files)-1):
        name1_core = os.path.splitext(name1)[0]
        name2_core = os.path.splitext(name2)[0]
        full_path1 = os.path.join(args.img_dir, name1)
        full_path2 = os.path.join(args.img_dir, name2)
        img1 = cv2.imread(os.path.join(args.img_dir, name1))
        img2 = cv2.imread(os.path.join(args.img_dir, name2))
        R, t, kpt0, kpt1 = step_optical(args, img1, img2)
        if args.viz:
            matched_fig = make_matching_plot(img1, img2, kpt0, kpt1)
            cv2.imwrite(os.path.join(args.output_fig_dir,"matching_{}_{}.png".format(name1_core, name2_core)), matched_fig)
        with open(os.path.join(args.output_pos_dir,"matching_{}_{}.txt".format(name1_core, name2_core)), 'w')as f:
            f.write("{:0.4f} {:0.4f} {:0.4f}".format(np.arccos(R[0,0]), t[0], t[1]))