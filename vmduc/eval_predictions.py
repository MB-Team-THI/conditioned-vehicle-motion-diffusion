from improved_diffusion.vmm import *
import pandas as pd
import numpy as np
import os 
import scipy
import argparse
from improved_diffusion.script_util import add_dict_to_argparser



def main():
    args = create_argparser().parse_args()
    
    result_dir = os.path.join(os.getcwd(), 'ckpts', args.run_name, 'results')
    path = os.path.join(result_dir, 'results_gencfg='+str(args.cfg_scale)+".pkl")
    print('Evaluate: ', path)
    dfd = pd.read_pickle(path)
    keys = args.keys
    mse_x = 0
    mse_y = 0
    mse=0
    fde =0
    for i in range(0, len(dfd)):
        #x_gt, y_gt =  np.array(dfd['x_gt'][i]), np.array(dfd['y_gt'][i])
        # x_pred, y_pred = np.array(dfd['x_pred'][i]), np.array(dfd['y_pred'][i])
        axp, dpsip = np.array(dfd['ax_pred'][i]), np.array(dfd['dpsi_pred'][i])
        vx0 = dfd['v0'][i]
        psi0 = dfd['psi_0'][i]
    
        # Ground Truth Future Trajectory from motion params
        ax_gt = scipy.io.loadmat(dfd['scenario_file'][i])[keys[0]][0]
        dpsi_gt = scipy.io.loadmat(dfd['scenario_file'][i])[keys[1]][0]
        sample = vmm(ax_gt, dpsi_gt, v_init=vx0, psi_init=psi0, dt = args.dt)
        x_gt, y_gt = sample[0], sample[1]
        
        # Pred Futures

        sample = vmm(axp, dpsip, v_init=vx0, psi_init=psi0, dt = args.dt)
        x_pred = sample[0]
        y_pred = sample[1]
        
       
        mse_x += np.sqrt(np.mean((x_gt-x_pred)**2))/(len(dfd))
        mse_y += np.sqrt(np.mean((y_gt-y_pred)**2))/(len(dfd))
        mse += np.sqrt(np.mean((np.hstack((x_gt, y_gt))-np.hstack((x_pred, y_pred)))**2))/(len(dfd))
        fde += np.sqrt(np.mean((np.hstack((x_gt[-1], y_gt[-1]))-np.hstack((x_pred[-1], y_pred[-1])))**2))/(len(dfd))
    
    print('mse: ', mse )
    print('fde: ', fde)

def create_argparser():
    defaults = dict(
        cfg_scale=-1,
        run_name = 'example' ,
        dt = 1.0/25,
        keys= ['predicted_ax', 'predicted_dpsi']
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

