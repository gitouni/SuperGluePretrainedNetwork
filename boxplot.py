import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",type=str,default="knife1")
    parser.add_argument("--gt_dir",type=str,default="res/slip/{name}/pos")
    parser.add_argument("--methods",type=str,nargs="+",default=['NS','TELEA','Palette'])
    parser.add_argument("--pred_dir_fmt",type=str,default="res/slip/{name}/{method}")
    parser.add_argument("--output_fmt",type=str,default="fig/{name}_cmp_ref.png")
    parser.add_argument("--fig_unit",type=str,nargs=3,default=['rad','pixel','pixel'])
    parser.add_argument("--sign",type=float,nargs=3,default=[1,1,1])
    return parser.parse_args()

def absolute_error(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.abs(x-y)

if __name__ == "__main__":
    args = options()
    gt_dir = args.gt_dir.format(name=args.name)
    gt_files = sorted(os.listdir(gt_dir))
    gt_data =[]
    for gt_file in gt_files:
        with open(os.path.join(gt_dir, gt_file),'r') as f:
            raw_data = f.readline().rstrip('\n')
            item = raw_data.split(" ")
            gt_data.append(item)
    gt_data = np.array(gt_data, dtype=np.float32)
    x0 = np.linspace(1, 12, 3)
    handles = []
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Yaw (rad)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('x/y (pixel)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    for i,method in enumerate(args.methods):
        pred_dir = args.pred_dir_fmt.format(name=args.name, method=method)
        pred_files = sorted(os.listdir(pred_dir))
        pred_data = []
        for pred_file in pred_files:
            with open(os.path.join(pred_dir, pred_file),'r') as f:
                raw_data = f.readline().rstrip('\n')
                item = raw_data.split(" ")
                pred_data.append(item)
        pred_data = np.array(pred_data, dtype=np.float32)
        y = absolute_error(gt_data, pred_data).T  # (3,N)
        box = ax1.boxplot(y[0].tolist(),positions=[x0[0]],patch_artist=True,showmeans=True,showfliers=False,widths=0.5,
            boxprops={"facecolor": "#FFFFFF00",
                      "edgecolor": "C%d"%i,
                      "linewidth": 1.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'x',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':7})
        handles.append(box['boxes'][0])
        ax2.boxplot(y[1:].tolist(),positions=[x0[1],x0[2]],patch_artist=True,showmeans=True,showfliers=False,widths=0.5,
            boxprops={"facecolor": "#FFFFFF00",
                      "edgecolor": "C%d"%i,
                      "linewidth": 1.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'x',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':7})
        x0 += 1
    plt.xticks(x0-2,['yaw','x','y'])
    plt.legend(handles=handles, labels = args.methods, loc='upper center',ncol=3)
    plt.tight_layout()
    plt.savefig(args.output_fmt.format(name=args.name))