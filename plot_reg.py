from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from utils.utils import relpos_to_displacement

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir",type=str,default="res/slip/knife1/pos")
    parser.add_argument("--pred_dir",type=str,default="res/slip/knife1/Palette")
    parser.add_argument("--output_fmt",type=str,default="fig/knife1_Palette_{}.png")
    parser.add_argument("--fig_label",type=str,nargs=3, default=['yaw','x','y'])
    parser.add_argument("--use_dis",type=bool,default=True)
    parser.add_argument("--fig_unit",type=str,nargs=3,default=['rad','pixel','pixel'])
    parser.add_argument("--sign",type=float,nargs=3,default=[1,1,1])
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    gt_files = sorted(os.listdir(args.gt_dir))
    pred_files = sorted(os.listdir(args.pred_dir))
    gt_data =[]
    pred_data = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        with open(os.path.join(args.gt_dir, gt_file),'r') as f:
            raw_data = f.readline().rstrip('\n')
            item = raw_data.split(" ")
            gt_data.append(item)
        with open(os.path.join(args.pred_dir, pred_file),'r') as f:
            raw_data = f.readline().rstrip('\n')
            item = raw_data.split(" ")
            pred_data.append(item)
    gt_data = np.array(gt_data, dtype=np.float32)  # (N, 3)
    pred_data = np.array(pred_data, dtype=np.float32)  # (N, 3)
    if not args.use_dis:
        for i, (label, unit, sign) in enumerate(zip(args.fig_label,args.fig_unit, args.sign)):
            gt_single_axis = sign * gt_data[:,i]
            pred_single_axis = pred_data[:,i]
            max_value = max(np.max(gt_single_axis), np.max(pred_single_axis))
            min_value = min(np.min(gt_single_axis), np.min(pred_single_axis))
            plt.figure()
            plt.plot([min_value,max_value],[min_value,max_value],'k--')
            plt.scatter(pred_single_axis, gt_single_axis,s=np.ones(len(gt_data))*25,alpha=0.5,c=np.random.rand(len(gt_data)))
            plt.xlabel("prediction/{}".format(unit))
            plt.ylabel('gt/{}'.format(unit))
            plt.savefig(args.output_fmt.format(label))
    else:
        gt_dis = relpos_to_displacement(gt_data)
        pred_dis = relpos_to_displacement(pred_data)
        max_value = max(np.max(gt_dis), np.max(pred_dis))
        min_value = min(np.min(gt_dis), np.min(pred_dis))
        plt.figure()
        plt.plot([min_value,max_value],[min_value,max_value],'k--')
        plt.scatter(pred_dis, gt_dis,s=np.ones(len(gt_data))*25,alpha=0.5,c=np.random.rand(len(gt_data)))
        plt.xlabel("prediction/{}".format('pixel'))
        plt.ylabel('gt/{}'.format('pixel'))
        plt.savefig(args.output_fmt.format('dis'))