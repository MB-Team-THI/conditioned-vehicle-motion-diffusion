import torch
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.sequential.model import SeqEncoder, SeqDecoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.losses.classification import ClassLoss
from taming.data.motion_dataset import NormDataset


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.sequential = ddconfig['sequential_model']
        if self.sequential:
            self.encoder = SeqEncoder(**ddconfig)
            self.decoder = SeqDecoder(**ddconfig)
        self.time_steps = ddconfig['time_steps']
        self.norm_data = NormDataset(ddconfig['norm_vals'])
        self.loss = instantiate_from_config(lossconfig)
        
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        if self.sequential:
            dec, c_hat = self.decoder(quant)
            return dec, c_hat
        else:
            dec = self.decoder(quant)
            return dec, None
        
    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, info, h_z = self.encode(input)
        dec, c_hat = self.decode(quant)
        q_idx = info[-1]
        return dec, diff, q_idx, quant, c_hat, h_z

    def get_input(self, batch, k):
        dataX = batch['observed_data_x'] 
        dataY = batch['observed_data_y']
        dataVX = batch['observed_data_vx']
        dataVY = batch['observed_data_vy']  
        scenario_id = batch['scenario_id']
        dataX, dataY, dataVX, dataVY = self.norm_data.norm_data(dataX, dataY, dataVX, dataVY)
        # Dimension: Batch X Vehicles X Frames X Feature-Channels: 8x9x75x4
        x = torch.stack((dataX, dataY, dataVX, dataVY), dim =-1)
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float(), scenario_id

    def training_step(self, batch, batch_idx):
        x, scenario_id = self.get_input(batch, self.image_key)
        c_vec = torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
        for idx in range(0, x.shape[0]):
            if 'lcr' in scenario_id[idx]:
                c_vec[idx] = 1 
            elif 'lcl' in scenario_id[idx]:
                c_vec[idx] = 2

        xrec, qloss, q_idx, z_q, c_hat, h_z = self(x)
        
        # autoencode
        [total_loss, rec_loss, cl_loss], log = self.loss(qloss, x, xrec, c_hat, c_vec,self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # total loss
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/quant_loss", qloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/cl_loss", cl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return total_loss


    def validation_step(self, batch, batch_idx):
        
        x, scenario_id = self.get_input(batch, self.image_key)
        c_vec = torch.zeros(x.shape[0], dtype =torch.long).to(self.device)
        for idx in range(0, x.shape[0]):
            if 'lcr' in scenario_id[idx]:
                c_vec[idx] = 1 
            elif 'lcl' in scenario_id[idx]:
                c_vec[idx] = 2
        xrec, qloss, q_idx, z_q, c_hat, h_z = self(x)
        # autoencode
        [total_loss, rec_loss, cl_loss], log = self.loss(qloss, x, xrec, c_hat, c_vec,self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        # total loss
        self.log("val/total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/quant_loss", qloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/cl_loss", cl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []


    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x, scenario_id = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _, _, _, _= self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
