# cVMD: conditioned Vehicle Motion Diffusion Model


[![arXiv](https://img.shields.io/badge/arXiv-<2405.14384>-<COLOR>.svg)](https://arxiv.org/abs/2405.14384)
[![VQSPEC](https://img.shields.io/badge/Embedding_Explorer-VQSPEC-blue)](https://mb-team-thi.github.io/VQSPEC/)



## Architecture
The official PyTorch implementation of the paper <br />
[**"Reliable Trajectory Prediction and Uncertainty Quantification with Conditioned Diffusion Models"**](https://arxiv.org/abs/2405.14384).
![CVMDarchitecutre](https://github.com/mariiilyn/test/assets/78954553/e6bfec67-80e4-4d22-af6d-cfa00ce41bdc)
[arXiv](https://arxiv.org/abs/2405.14384) | [BibTeX](#bibtex) 




## News

📢 **30/May/24** - Release of the official cVMD PyTorch implementation. Currently only supports CPU processing, GPU processing support available soon.

## Getting started

This code was tested on `Ubuntu 22.04.4 LTS` and requires:

* Python 3.8.5
* conda 
* (soon: CUDA capable GPU (one is enough))<br />

### 1. Setup environment

Install the OpenMPI development libraries:
```shell
sudo apt-get install libopenmpi-dev
```

Setup conda env:
```shell
conda env create -f environment.yml
conda activate cvmd
```
<br />

### 2. Get data
The official implementation was tested on the highD dataset. However, also other publicly available datasets can be used.

NOTE: Please be aware that the data examples in this repository are not taken from the highD or any other dataset. Instead, they are computer-generated for illustrative purposes. The features may not necessarily reflect actual, natural vehicle behavior.

<details>
  <summary><b>highD</b></summary>

To get the data follow the steps provided on the dataset homepage: https://levelxdata.com/highd-dataset/.

</details>
<details>
  <summary><b>NGSIM</b></summary>

To get the data follow the steps provided on the dataset homepage: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm.

</details>
<details>
  <summary><b>Automatum data</b></summary>

To get the data follow the steps provided on the dataset homepage: https://automatum-data.com/de.

</details>
<br />

### 3. Prepare Data
Within the `data` folder generate a new folder for each dataset, e.g. highD.
```
├── data
│   ├── example_dataset
│   ├── highD
│   ├── ...
```
Each `dataset` subfolder needs to follow this hierarchical structure
```
data/highD/
├── train
│   ├── class0
|   │   ├── kl0.mat
|   │   ├── ...
│   ├── class1
|   │   ├── lcr0.mat
|   │   ├── ...
│   ├── class2
|   │   ├── lcl0.mat
|   │   ├── ...
├── test
│   ├── class0
|   │   ├── ...
│   ├── class1
|   │   ├── ...
│   ├── class2
|   │   ├── ...
├── ...
```
* `class0` Holds scenarios classifies as keep lane (kl) scenarios.
* `class1` Holds scenarios classifies as lange change right (lcr) scenarios.
* `class2` Holds scenarios classifies as lange change left  (lcl) scenarios.


xxx.mat file structure

```
xxx.mat
  |    - keys -                          - type -                 - size -
  ├── 'data_keys'                        Array of strings          1 x N
  └── 'observed_data_x'                  Array of float64          N x T_o
  └── 'observed_data_y'                  Array of float64          N x T_o
  └── 'observed_data_vx'                 Array of float64          N x T_o
  └── 'observed_data_vy'                 Array of float64          N x T_o
  └── 'scenario_type'                    Array of strings          1 x 1
  └── 'predicted_x'                      Array of float64          1 x T_p
  └── 'predicted_y'                      Array of float64          1 x T_p
  └── 'predicted_ax'                     Array of float64          1 x T_p
  └── 'predicted_dpsi'                   Array of float64          1 x T_p
  └── 'psi_0'                            Array of float64          1 x 1
  └── 'v0'                               Array of float64          1 x 1
```
Dimensions:
* `N` - Maximal number of considered vehicles (incl. ego).
* `T_o` - Number of observation steps (t_obs x f, e.g : 3s x 25 Hz = 75 steps).
* `T_p` - Number of prediction steps (t_pred x f, e.g : 5s x 25 Hz = 125 steps).
  
```
'data_keys' = ['ego ', 'following ', 'preceding',
       'leftPreceding ', 'leftAlongside ', 'leftFollowing ',
       'rightPreceding', 'rightAlongside', 'rightFollowing'] 
```
```
'scenario_type' = ['keep_lane']  or  ['lane_change_left']  or ['lane_change_right']
```



  
<br />

### 4. Train model

The training processes of the (1) context conditioning module and the (2) vehicle motion diffusion module within cVMD are decoupled. 
The adaptive guidance scale computation (uncertainty quantification) is performed within the vehicle motion diffusion module.

<details>
  <summary><b> (1) Context Conditioning (VQ-VAE)</b></summary>
  
#### 1 ) Set hyperparameters
To train the VQ-VAE Context Conditioning model, you should first decide some hyperparameters definded within the config file `vqvae/configs/custom_vqvae.yaml`.
Parameters can be overwritten or added with command-line options of the form `--key value`.

#### 2 ) Run training loop
Change directory to folder `vqvae`:
```shell
cd vqvae
```
Then run
```shell
python main.py --base ./configs/custom_vqvae.yaml
```

* Use `--base` to define the path to the hyperparameter config file.

**You may also define:**
* Use `--resume` path of model checkpoint to resume from.
* Use `--postfix` post-postfix for default name.
* Use `--seed` set seed for seed_everything.
* Use `--debug` enable post-mortem debugging.
* Use `--gpus` to define on a how many GPUs to train. (Not yet supported)

**Running those will get you:**
* `vqvae/logs/run_name_vq` folder containining saved checkpoints (top K=3) and logged training information.

#### 3 ) MLE 
Apply Maximum Likelihood Estimation (MLE) to find the most likely parameters for the conditional Gaussian distributions that best fit the underlying latent variables generated by the VQ-VAE (based on train data). This step is required for the uncertainty quantification associated with a new, unseen scenario. 
If not done before, change directory to folder `vqvae`:
```shell
cd vqvae
```
Then run
```shell
python main_inference.py --resume_ckpt "./logs/run_name_vq/checkpoints/epoch=xxx.ckpt"
```
* Use `--resume_ckpt` path of model checkpoint to resume from.
  
**You may also define:**
* Use `--pkl_name` prefix for the resulting embedding .pkl-files.
* Use `--result_dir` postfix for logdir where results are stored (default: vqresults).
* Use `--seed` set seed for seed_everything.
* Use `--plot_hist` If set to True a histogram for the codebook entry assignment is generated.

**Running those will get you:**
* `vqresults/example_run/epoch=xxx` folder containining the following information:
  * `meta_codebook_train.pkl` codebook information after training and MLE.
  * `meta_embeddings_train.pkl` embedding information for the train data generated by the VQ-VAE.
  * `meta_embeddings_test.pkl` embedding information for the test data generated by the VQ-VAE.
</details>

<details>
  <summary><b> (2) Vehicle Motion Diffusion</b></summary>

#### 1 ) Run training loop
Change directory to folder `vmduc`:
```shell
cd vmduc
```
Then run
```shell
python main.py --data_dir "../data/highD/train" --vqvae_dir "../vqresults/run_name_vq/epoch=xxx"
```
* Use `--data_dir` to define the path to the dataset.
* Use `--vqvae_dir` to define the path to the resulting embeddings of the VQ-VAE.

**You may also define:**
* Use `--batch_size` to define the batch size.
* Use `--lr` to define the learning rate.
* Use `--emb_train` to define the name of the generated embeddings-information file (train data) to be loaded.
* Use `--meta_codebook` to define the name of the generated codebook-information file to be loaded.
* Use `--log_interval` to define the log interval.
* Use `--save_interval` to define the save interval.
* ...

**Running those will get you:**
* `vmduc/ckpts/run_name_vmd` folder containining saved model checkpoints and logged training information.


</details>
<br />

### 5. Test the trained model
If not done before, change directory to folder `vmduc`:
```shell
cd vmduc
```
Then run
```shell
python main_test.py --model_path "./ckpts/run_name_vmd/model=yyy.pt" --vqvae_dir "../vqresults/run_name_vq/epoch=xxx"
```
<br />
* Use `--model_path` to define the path to the VMD-model to be loaded.
* Use `--vqvae_dir` to define the path to the resulting embeddings of the VQ-VAE.


**You may also define:**
* Use `--cfg_scale` to define the guidance scale. If set to -1, the guidance scale is computed adaptively.
* Use `--data_dir` to define the path to the test dataset.
* Use `--emb_test` to define the prefix of the generated embeddings-information files to be loaded.
* Use `--meta_codebook` to define the name of the generated codebook-information file to be loaded.
* Use `--wmin` to define the minimal adaptive guidance scale.
* Use `--wmax` to define the maximal adaptive guidance scale.
* Use `--mth` to define the threshold for the maximal M-distance.
* ...

**Running those will get you:**
* `vmduc/ckpts/run_name_vmd/results/results_gencfg=Z.pkl` file containing the generated predictions and ground truth information.

<br />

### 6. Evaluate
If not done before, change directory to folder `vmduc`:
```shell
cd vmduc
```
Then run
```shell
python main_test.py --model_path "./ckpts/run_name_vmd/model=yyy.pt" --vqvae_dir "../vqresults/run_name_vq/epoch=xxx" --dt 1.0/25.0
```

* Use `--model_path` to define the path to the VMD-model to be loaded.
* Use `--vqvae_dir` to define the path to the resulting embeddings of the VQ-VAE.
* Use `--dt` to define sampling interval (dt = 1 / f, where f = sample rate of dataset).

<br />

## Acknowledgments

This code builds upon the impactful work of predecessors. We want to thank the following contributors
that our code is based on:
[improved-diffusion](https://github.com/openai/improved-diffusion), [VQ-GAN](https://github.com/CompVis/taming-transformers)

<br />

## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that the code depends on other libraries and use a dataset that each have their own respective licenses that must also be followed.

<br />

## Bibtex
If you find this code useful in your research, please cite:
```
@inproceedings{
neumeier2024cvmd,
title={Reliable Trajectory Prediction and Uncertainty Quantification with Conditioned Diffusion Models},
author={Marion Neumeier and Sebastian Dorn and Michael Botsch and Wolfgang Utschick},
booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
year={2024},
url={}
}
```

