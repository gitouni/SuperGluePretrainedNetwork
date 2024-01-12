import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RANSACRegressor
from models.utils import make_matching_plot
import cv2


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


if __name__ == "__main__":
    data = np.load("res/card/pos/matches_000021_000022.npz")
    img1 = cv2.imread("data/card/Palette/Out_0021.bmp")
    img2 = cv2.imread("data/card/Palette/Out_0022.bmp")
    kpt0 = data['kpt0']  # (N,2)
    kpt1 = data['kpt1']  # (N,2)
    base_estimator = SolveEstimator()
    base_estimator.fit(kpt0, kpt1)
    param = base_estimator.get_params()
    R = param['rot']
    t = param['tsl']
    print("Oridnary Solution:\nR:{}\nt:{}".format(R,t))
    print("RSME:{}".format(base_estimator.score(kpt0, kpt1)))
    ransac_estimator = RANSACRegressor(estimator=base_estimator, min_samples=5, random_state=0, residual_threshold=5)
    ransac_estimator.fit(kpt0, kpt1)
    valid = ransac_estimator.inlier_mask_
    param = ransac_estimator.estimator_.get_params()
    R = param['rot']
    t = param['tsl']
    print("Inlier | Total: {}, {}".format(valid.sum(), valid.size))
    print("RANSAC Solution:\nR:{}\nt:{}".format(R,t))
    print("RSME:{}".format(ransac_estimator.score(kpt0, kpt1)))
    print("Inlier RMSE:{}".format(ransac_estimator.score(kpt0[valid], kpt1[valid])))
    out1 = make_matching_plot(img1, img2, kpt0, kpt1)
    out2 = make_matching_plot(img1, img2, kpt0[valid], kpt1[valid])
    out3 = make_matching_plot(img1, img2, kpt0[~valid], kpt1[~valid], color=np.array([0,0,255]*len(kpt0)).reshape(-1,3))
    cv2.imwrite("bfo_ransac.png",out1)
    cv2.imwrite("aft_ransac.png",out2)
    cv2.imwrite("err_corr.png",out3)
    homography, mask = cv2.findHomography(kpt0.reshape(-1,1,2), kpt1.reshape(-1,1,2), cv2.RANSAC, 5.0)
    mask = mask.reshape(-1).astype(np.bool_)
    out4 = make_matching_plot(img1, img2, kpt0[mask], kpt1[mask])
    result = cv2.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))
    # Blending the warped image with the second image using alpha blending
    alpha = 0.5  # blending factor
    blended_image = cv2.addWeighted(result, alpha, img2, 1 - alpha, 0)
    # Display the blended image
    cv2.imwrite('blended.png', blended_image)
    cv2.imwrite("aft_homo_ransac.png",out4)