#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:09:08 2023

@author: blacksmurf
"""

import torch as th
MAX_ACC = 4.0
MAX_dPSI = 1.0
MAX_X = 150.0
MAX_Y = 10.0

## NORM Vehicle Motion Model
#-------------------------------------------------------------------------------------
def vmm_norm(acc, d_psi):
    # keys =  ['ego_x', 'ego_y', 'ego_psi', 'ego_vx', 'ego_vy', 'ego_dpsi', 'ego_ax', 'ego_ay']
    flag = False
    
    if th.max((d_psi)) > MAX_dPSI:
        flag = True
    if th.min(d_psi) < -MAX_dPSI:
        flag = True
    
    norm_acc = (acc / MAX_ACC).clamp(-1, 1)
    norm_dpsi = (d_psi / MAX_dPSI).clamp(-1,1)
    batch_norm = th.cat((norm_acc, norm_dpsi),dim=1)
    return batch_norm, flag

def inv_vmm_norm(batch):
    batch[:, 0, :] = batch[:,0,:]*MAX_ACC
    batch[:, 1, :] = batch[:,1,:]*MAX_dPSI
    return batch
    
## NORM XY Prediction
#-------------------------------------------------------------------------------------
def xy_norm(batch):    
    # keys = [ .... ]
    batch = th.cat((batch[0], batch[1]),dim=1)
    batch[:,0,:] =  ((batch[:,0,:] -MAX_X)/MAX_X).clamp(-1, 1)
    batch[:,1,:] =  ((batch[:,1,:])/MAX_Y).clamp(-1, 1)
    return batch

def inv_xy_norm(batch, type_ ='xy'):
    if type_ == 'xy':
        batch[:,0,:] =  ((batch[:,0,:] + 1)*MAX_X)
        batch[:,1,:] =  ((batch[:,1,:])*MAX_Y)
    return batch    


