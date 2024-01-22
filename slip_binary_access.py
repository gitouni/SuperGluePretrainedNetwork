from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from utils.utils import relpos_to_displacement

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj",type=str,default='screwdriver1')
    parser.add_argument("--methods",type=str,nargs="+",default=['None','NS','TELEA','Palette'])
    parser.add_argument("--gt_dir",type=str,default="SPG/{obj}/markerless")
    parser.add_argument("--pred_dir",type=str,default="SPG/{obj}/{method}")
    parser.add_argument("--output_fig",type=str,default="res/slip/{obj}/slip_cmp.pdf")
    parser.add_argument("--output_res",type=str,default="res/slip/{obj}/slip_cmp.txt")
    parser.add_argument("--gt_slip_threshold",type=bool,default=1.0)
    parser.add_argument("--pred_slip_threshold",type=bool,default=1.0)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    gt_dir = args.gt_dir.format(obj=args.obj)
    gt_files = sorted(os.listdir(gt_dir))
    gt_data =[]
    for gt_file in gt_files:
        with open(os.path.join(gt_dir, gt_file),'r') as f:
            raw_data = f.readline().rstrip('\n')
            item = raw_data.split(" ")
            gt_data.append(item)
    gt_data = np.array(gt_data, dtype=np.float32)  # (N, 3)
    gt_dis = relpos_to_displacement(gt_data)
    gt_slip = gt_dis > args.gt_slip_threshold
    plt.subplot(5,1,1)
    plt.plot(gt_dis)
    plt.title('gt')
    acc = dict()
    for i,method in enumerate(args.methods):
        pred_dir = args.pred_dir.format(obj=args.obj, method=method)
        pred_files = sorted(os.listdir(pred_dir))
        pred_data = []
        for pred_file in pred_files:
            with open(os.path.join(pred_dir, pred_file),'r') as f:
                raw_data = f.readline().rstrip('\n')
                item = raw_data.split(" ")
                pred_data.append(item)
        pred_data = np.array(pred_data, dtype=np.float32)  # (N, 3)
        pred_dis = relpos_to_displacement(pred_data)
        
        pred_slip = pred_dis > args.pred_slip_threshold
        correct = gt_slip==pred_slip
        acc[method] = correct.sum()/len(gt_slip)
        err_idx = np.arange(len(gt_slip))[~correct]
        plt.subplot(5,1,i+2)
        plt.plot(pred_dis)
        plt.plot(err_idx, pred_slip[~correct], 'r*')
        plt.title(method)
    plt.tight_layout()
    plt.savefig(args.output_fig.format(obj=args.obj))
    with open(args.output_res.format(obj=args.obj), 'w') as f:
        for key, value in acc.items():
            f.write("{}: {:.2%}\n".format(key, value))
    print(acc)
    # max_value = max(np.max(gt_dis), np.max(pred_dis))
    # min_value = min(np.min(gt_dis), np.min(pred_dis))
    # plt.figure()
    # plt.plot([min_value,max_value],[min_value,max_value],'k--')
    # plt.scatter(pred_dis, gt_dis,s=np.ones(len(gt_data))*25,alpha=0.5,c=np.random.rand(len(gt_data)))
    # plt.xlabel("prediction/{}".format('pixel'))
    # plt.ylabel('gt/{}'.format('pixel'))
    # plt.savefig(args.output_fmt.format('dis'))