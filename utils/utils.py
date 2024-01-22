import shutil, os
import cv2
import numpy as np
from typing import Tuple

def refresh_dir(dirname:str):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def relpos_to_displacement(rel_pos_list:np.ndarray) -> np.ndarray:
    rel_displace = []
    last_pos = np.eye(3)
    for rel_pos in rel_pos_list:
        th, x, y = rel_pos
        rel_pos_matrix = np.array([[np.cos(th), -np.sin(th), x],
                                   [np.sin(th), np.cos(th), y],
                                   [0,0,1]])
        cur_pos = np.matmul(rel_pos_matrix, last_pos)
        rel_dis = np.sqrt((cur_pos[0,2]-last_pos[0,2])**2 + (cur_pos[1,2]-last_pos[1,2])**2)
        rel_displace.append(rel_dis)
        last_pos = np.copy(cur_pos) # copy not reference
    return np.array(rel_displace)

def coord_tran(coord:list,kx:float,ky:float):
    if abs(kx-1) < 1e-4 and abs(ky-1) < 1e-4:
        return coord
    new_coord = []
    assert(len(coord) == 2 or len(coord) == 4), "coord size must be 2 or 4!"
    if len(coord) == 2:
        new_coord.append(coord[0] * kx)
        new_coord.append(coord[1] * ky)
    else:
        new_coord.append(coord[0] * kx)
        new_coord.append(coord[1] * ky)
        new_coord.append(coord[2] * kx)
        new_coord.append(coord[3] * ky)
    return new_coord

def reverse_size(size:list):
    assert(len(size) == 2), "size must has 2 elements"
    return size[1], size[0]

def detect_keypoints(img:np.ndarray, mask:np.ndarray, maxCorners:int, qualityLevel:float, minDistance:float, blockSize:int) -> np.ndarray:
    if np.ndim(img) == 3:
        _img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        _img = img
    p0 = cv2.goodFeaturesToTrack(_img, maxCorners, qualityLevel, minDistance, mask=mask, blockSize=blockSize)
    return np.squeeze(p0,axis=1)  # (N, 1, 2) -> (N, 2)

def track_keypoints(src:np.ndarray,tgt:np.ndarray, p0:np.ndarray, winSize:Tuple[int, int], maxLevel:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.ndim(src) == 3:
        _src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        _src = src
    if np.ndim(tgt) == 3:
        _tgt = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    else:
        _tgt = tgt
    p1, st, err = cv2.calcOpticalFlowPyrLK(_src, _tgt, p0, None, winSize=winSize, maxLevel=maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return p1, st.flatten(), err.flatten()

def detect_track_keypoints(src:np.ndarray,tgt:np.ndarray,
         mask:np.ndarray, maxCorners:int, qualityLevel:float, minDistance:float, blockSize:int,
         winSize:Tuple[int, int], maxLevel:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if np.ndim(src) == 3:
        _src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        _src = src
    if np.ndim(tgt) == 3:
        _tgt = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    else:
        _tgt = tgt
    p0 = detect_keypoints(_src, mask, maxCorners, qualityLevel, minDistance, blockSize)
    p1, st, err = cv2.calcOpticalFlowPyrLK(_src, _tgt, p0, None, winSize=winSize, maxLevel=maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    return p0, p1, st.flatten(), err.flatten()

def justify_keypoints(src:np.ndarray,tgt:np.ndarray, p0:np.ndarray, p1:np.ndarray, winSize:Tuple[int, int], maxLevel:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.ndim(src) == 3:
        _src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        _src = src
    if np.ndim(tgt) == 3:
        _tgt = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    else:
        _tgt = tgt
    p1, st, err = cv2.calcOpticalFlowPyrLK(_src, _tgt, p0, p1, winSize=winSize, maxLevel=maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    return p1, st.flatten(), err.flatten()