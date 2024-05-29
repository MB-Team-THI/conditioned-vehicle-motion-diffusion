import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
import torch
import scipy
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import pandas as pd
from taming.util import *

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload: 
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--pkl_name",
        type=str,
        const=True,
        default="meta_embeddings",
        nargs="?",
        help="prefix for the resulting embedding files",
    )
    parser.add_argument(
        "-d",
        "--result_dir",
        type=str,
        const=True,
        default="vqresults",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume_ckpt",
        type=str,
        const=True,
        default="/home/blacksmurf/Desktop/Marion/VMD/cvmd_github/vqvae/logs/example/checkpoints/epoch=000052.ckpt",
        nargs="?",
        help="resume from checkpoint in logdir",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed forseed_everything",
    )
    parser.add_argument(
        "-p",
        "--plot_hist",
        type=bool,
        default=False,
        help="plot histogram.",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def save_results(df, sdir, file_name, mode):
    if not os.path.exists(sdir):
        os.makedirs(sdir)       
    save_path  =  os.path.join(sdir, file_name + '_' + mode)
    df.to_pickle(save_path + '.pkl')
    print('Results stored to' + save_path)
    return sdir
       

if __name__ == "__main__":


    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.resume_ckpt:
        if not os.path.exists(opt.resume_ckpt):
            raise ValueError("Cannot find {}".format(opt.resume_ckpt))
        if os.path.isfile(opt.resume_ckpt):
            paths = opt.resume_ckpt.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume_ckpt
            model_name = paths[-1].replace(".ckpt", "")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs#+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
        sdir =  os.path.join(os.getcwd(), "..", opt.result_dir, nowname, model_name)
    else:
        raise ValueError(
            "Define resume checkpoint --resume_ckpt"
        )

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        config.model.params.ckpt_path = opt.resume_ckpt
        
        #  load model 
        model = instantiate_from_config(config.model)
        model.eval()
        
        # load data
        config.data.params.batch_size =1
        data = instantiate_from_config(config.data)  
        data.setup("")
        
        # train data embedding
        dataloader = data.train_dataloader            
        with torch.no_grad():
            for r, datamode in enumerate(['train','test']):
                print('Inference on Data: ' + datamode)
                dataloader = data.train_dataloader if datamode == 'train' else data.test_dataloader
                idx_count = np.zeros([config.model.params.n_embed])
                df = pd.DataFrame(columns=['scenario_id', 'q_idx', 'q_vec', 'z_vec', 'l2norm'])
                for i, batch in enumerate(dataloader()):
                    dataX = batch['observed_data_x'] 
                    dataY = batch['observed_data_y'] 
                    dataVX = batch['observed_data_vx'] 
                    dataVY = batch['observed_data_vy']   
                    scenario_id = batch['scenario_id']
                    dataX, dataY, dataVX, dataVY = model.norm_data.norm_data(dataX, dataY, dataVX, dataVY) 
                    x = torch.stack((dataX, dataY, dataVX, dataVY), dim=1).float()
                    dec, diff, q_idx, quant, c, z_vec = model(x)
                    q_idx = q_idx.detach().numpy()[0]
                    new_entry = {'scenario_id': scenario_id[0],
                                 'q_idx':q_idx,
                                 'q_vec':quant[0,:,0,0].detach().numpy(),
                                 'z_vec':z_vec[0,:,0,0].detach().numpy(),
                                 'l2norm': diff.detach().numpy()} 
                    df.loc[len(df)] = new_entry
                    idx_count[q_idx] += 1
                    if datamode == 'test':
                        if i == 0:
                            dfm = pd.DataFrame(columns=['scenario_id', 'mdist'])
                        if meta_codebook['q_bool'][q_idx]:
                            u = meta_codebook['q_mu'][q_idx]
                            S = meta_codebook['q_cov'][q_idx]
                            v = z_vec[0,:,0,0].detach().numpy()
                            mdist = scipy.spatial.distance.mahalanobis(u,v,S)
                        else:
                            mdist = -1
                        md_entry = {'scenario_id': scenario_id[0],
                                     'mdist': diff.detach().numpy()} 
                        dfm.loc[len(dfm)]= md_entry
            
                if datamode =='train':
                    meta_codebook = compute_mucov(df, idxs=config.model.params.n_embed)
                    save_results(meta_codebook, sdir, 'meta_codebook', datamode)
                    # df = pd.merge(df, meta_codebook, on='q_idx', how='outer').dropna()
                else:
                    df = pd.merge(df, dfm, on='scenario_id', how='outer')
                      
                save_results(df, sdir, opt.pkl_name, datamode)
                lcr, kl, lcl= get_histinfo(df.copy(), config.model.params.n_embed, sdir, datamode)
                compute_impurity(lcr, kl, lcl, datamode)
                if opt.plot_hist:
                    plot_idx_dist(df.copy(), bins=config.model.params.n_embed)
            
    except Exception:
        raise


