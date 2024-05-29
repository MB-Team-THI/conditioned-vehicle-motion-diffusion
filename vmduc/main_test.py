import argparse
import os
import pandas as pd
import numpy as np
import torch as th
from improved_diffusion.image_datasets import load_data_once

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.norm_utils import *

def mdist2w(dist, wmin=1, wmax=7, mth=10):
    tc = np.clip(dist, 0, mth) / mth
    w = wmin + (1-tc) * (wmax-wmin)
    return w.reshape(-1)

def set_path(model_path, folder_name='results'):
    dir_name = os.path.dirname(model_path)
    out_dir = os.path.join(dir_name, folder_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def main():
    args = create_argparser().parse_args()
    out_dir = set_path(args.model_path)
    dist_util.setup_dist()
    logger.configure(out_dir)

    vqdf = pd.read_pickle(os.path.join(args.vqvae_dir, args.emb_test))
    codebook_path = os.path.join(args.vqvae_dir, args.meta_codebook)
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(codebook_path = codebook_path, cfg_scale=args.cfg_scale,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if diffusion.cfg_scale == -1:
        args.batch_size = 1 
        model, diffusion = create_model_and_diffusion(codebook_path = codebook_path, cfg_scale=args.cfg_scale,
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    
    meta_codebook = pd.read_pickle(codebook_path)
    data = load_data_once(data_dir=args.data_dir, batch_size=args.batch_size, class_cond=args.class_cond,)
    cfgs = diffusion.cfg_scale
    df = pd.DataFrame(columns=['scenario_file', 'q_idx',  'x_gt', 'y_gt'])
    for i, [batch, cond] in enumerate(data):
        model_kwargs = {}
        file = batch['file']
        q_idx = vqdf[vqdf['scenario_id'].isin(batch['scenario_id'])]['q_idx'].values
        x_gt = batch['predicted_x'][:,0,:]
        y_gt = batch['predicted_y'][:,0,:]
        psi0 = batch['psi_0'][:,0,0].reshape(-1,1)
        vx0 = batch['vx0'].reshape(-1,1)
        
        model_kwargs["y"] = th.Tensor(q_idx).long()
        if cfgs == -1:
            if not meta_codebook['q_bool'][q_idx].values:
                d = args.mth
            else:
                d = vqdf[vqdf['scenario_id'].isin(batch['scenario_id'])]['mdist'].values[0]
            w = th.Tensor(mdist2w(d, wmin=args.wmin, wmax=args.wmax, mth=args.mth))
            diffusion.cfg_scale = w
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size, 2, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        if model.use_vmm:
            sampleN = inv_vmm_norm(sample)
        else:
            sample = inv_xy_norm(sample)
        for i in range(args.batch_size):
            ndf = pd.DataFrame({
                'scenario_file': file[i], 
                'q_idx': q_idx[i], 
                'x_gt': [np.array(x_gt[i,:])], 
                'y_gt': [np.array(y_gt[i,:])], 
                'ax_pred': [sampleN[i,0,:].numpy()],
                'dpsi_pred': [sampleN[i, 1, :].numpy()],
                'v0': vx0.numpy()[i],
                'psi_0': psi0.numpy()[i]
            })
            df = pd.concat([df, ndf], ignore_index=True)
    out_path = os.path.join(logger.get_dir(), "results_gencfg="+str(cfgs))
    df.to_pickle(out_path +'.pkl') 


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=8,
        batch_size=8,
        use_ddim=False,
        cfg_scale=-1,
        wmin=1,
        wmax=7,
        mth=10,
        model_path= os.path.join(os.getcwd(), 'ckpts', 'example', 'ema_0.9999_000000.pt'),
        vqvae_dir= os.path.join(os.getcwd(), "..", "vqresults", "example", "epoch=000052"),
        data_dir= os.path.join(os.getcwd(), "..", "data" +os.sep +"highD" + os.sep +"test"),
        emb_test = 'meta_embeddings_test.pkl',
        meta_codebook = 'meta_codebook_train.pkl',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
