# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:15:34 2022

@author: bjh3
"""


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift
from . import ChiSq as CS
from .Classes import *
import h5py
from tqdm import tqdm
import random

ProjThFull = np.array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0., -15., -13., -11.,  -9.,  -7.,  -5.,  -3.,
        -1.,   1.,   3.,   5.,   7.,   9.,  11.,  13., 15.], dtype=np.float32)

ProjPsiFull = np.array([-14., -12., -10.,  -8.,  -6.,  -4.,  -2.,   0.,   2.,   4.,   6.,
         8.,  10.,  12.,  14.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 0.], dtype=np.float32)


def MakePhantom(fls, svfl, mdims = (3,512,128,128), dimscatt = (128,128), 
                ProjTh = ProjThFull, ProjPsi = ProjPsiFull, 
                x0 = 64.2, y0 = 64.6, bmwd = 1.1, rblock = 9., Nk = 100,
                fileformat = 'UberMag', CR = 1.5e6, bkg = 0.35, 
                Nz = np.array([1,4,4])*430, reducedim = 1, M0 = 1900, H0 = 250,
                Q0 = -0.75, h = np.array([-0.75,0,0]),
                wavelength = 6e-8, dx = 14.7E-7, 
                xFd = np.array(None), Fd = np.array(None), 
                Paxis = np.array(None), PsiSwitch = 45, ThetaSwitch = 45):
    bkg *= 1/CR

    cent = np.array([x0,y0,bmwd])
    
    fI00 = np.full(dimscatt,0.,dtype = np.float32)
    fI00[int(x0),int(y0)] = 1.
    fI00 = gaussian_filter(fI00,bmwd)
    fI00 = np.clip(shift(fI00,(x0 - int(x0),y0-int(x0))),0,None)
    
    mdimschi = (mdims[0],) + (int(mdims[1]/reducedim),) + mdims[2:]
    
    X = CS.PropParams(mdimschi, ProjTh, ProjPsi, fI00, Nz, wavelength, 
                 dx, xFd, Fd, dimscatt, Paxis, PsiSwitch, ThetaSwitch,
                 [reducedim,1,1])

    savg = np.full(ProjPsi.shape + dimscatt,0.,dtype = np.float32)
    
    for k in tqdm(range(Nk)):
    
        rll = (np.random.randint(0,mdims[1]),
               np.random.randint(0,mdims[2]),
               np.random.randint(0,mdims[3]))
        
        fl = random.choice(fls)
        hf = h5py.File(fl,'r')
        if fileformat == 'UberMag':
            mv = -np.roll(hf['field/array'][:].transpose((3,2,0,1)),1,axis=0)
        else:
            mv = hf['m'][()]
        hf.close()
        mv = np.roll(mv,rll,axis=(1,2,3))
        
        if reducedim == 4:
            mv = (mv[:,0::4,:,:] + mv[:,1::4,:,:] + mv[:,2::4,:,:] + mv[:,3::4,:,:])/4
        
        elif reducedim == 2:
            
            mv = (mv[:,0::2,:,:] + mv[:,1::2,:,:])/2
        
        S = SpinDensity(mv, M0, H0, [reducedim,1,1])
        S.initF(Q0, h)
        
        
        s, ss = CS.Fwrd(X,S,False)
        savg += s
    
    savg *= 1/Nk


    qx = np.linspace(0,dimscatt[0]-1,dimscatt[0])
    rr = (qx[:,None] - x0)**2 + (qx[None,:]-y0)**2
    R0Msk = np.sign(np.clip(rr - rblock**2,0,None))

    dQ = 2 * np.pi / (dx * 1e8) / dimscatt[0]
    Qx = qx[:,None] * dQ + qx[None,:] * 0
    Qy = qx[:,None] * 0 + qx[None,:] * dQ

    fI0 = np.random.poisson(fI00*CR)
    fIm = np.random.poisson((bkg+savg)*R0Msk*CR)
    
    sv = h5py.File(svfl,'a')
    sv.create_dataset('s', shape = savg.shape, dtype = np.float32, data = savg)
    sv.create_dataset('fI00', shape = fI00.shape, dtype = np.float32, data = fI00)
    sv.create_dataset('EmptBm', shape = fI00.shape, dtype = np.float32, data = fI0)
    sv.create_dataset('EmptBmUnc', shape = fI00.shape, dtype = np.float32, data = fI0**0.5)
    sv.create_dataset('SANS', shape = fIm.shape, dtype = np.float32, data = fIm)
    sv.create_dataset('SANSUnc', shape = fIm.shape, dtype = np.float32, data = fIm**0.5)
    sv.create_dataset('ProjPsi', shape = ProjPsi.shape, dtype = np.float32, data = ProjPsi)
    sv.create_dataset('ProjTh', shape = ProjTh.shape, dtype = np.float32, data = ProjTh)
    sv.create_dataset('cent', shape = cent.shape, dtype = np.float32, data = cent)
    sv.create_dataset('Qx', data = Qx)
    sv.create_dataset('Qy', data = Qy)
    sv.close()