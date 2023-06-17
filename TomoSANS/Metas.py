# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:13:34 2022

@author: bjh3
"""

import numpy as np
import h5py
from .Classes import *
from . import MovieMaker as MM
from . import SkyVis as SV
import matplotlib.pyplot as plt

def RunMetas(MSFRFile, Phantom = True, h= np.array([-0.75,0,0]), Hist = False, rtn = False, Movie = 'False', sv = None):
    
    
    hf = h5py.File(MSFRFile,'r')
    
    if Phantom:
        
        Qr = -0.759
        S = SpinDensity(-np.roll(hf['field/array'][:].transpose((3,2,0,1)),1,axis=0), 1900, 250, [1,1,1], 14.7e-7)
        S.initF(Qr,h)
    
    else:
        
        #Qr = -0.74246
        Qr = hf['Q0'][()]
        S = SpinDensity(hf['m'][:], hf['M0'][()], hf['H0'][()], hf['voxelaspect'][()], hf['dx'][()])
        S.initF(Qr,hf['h'][()])
        
    hf.close()
    
     
    
    
    
    if sv == 'auto':
    
        sv = MSFRFile.split('.')[0]
    
    if Phantom:
        
        Sns, Smz, SB, SS = PkHist(S, vmax = 30, rescale = 0.5, Hist = Hist, sv = sv)
        
    else:
    
        Sns, Smz, SB, SS = PkHist(S, vmax = 30, Hist = Hist, sv = sv)
    
    if Movie != 'False':
        
        if Movie == 'auto':
            
            MovieFile = MSFRFile.split('.')[0] + '.avi'
            MM.AnimateSCRGB(MovieFile, S)
            
        else:
            
            MM.AnimateSCRGB(Movie, S)
        
    
    if rtn:
        return S


def PkHist(S, Hist = True, rescale = 1, vmax = 60, sv = None):
    
    if not hasattr(S,'rhoEm'):
        
        S.ComputeMetas()
        
    if not hasattr(S,'locs'):
        
        S.MonopoleList()
    
    #pks = np.array(np.nonzero(PkMx(S.rhoEm**2,0.02**2)))
    
    #EmCh = PkAttr(pks, S.rhoEm)
    #dzMz = PkAttr(pks, S.dm[0,0])

    if Hist:    
        fig = SV.MakeScatt(S.EmCh, S.dz, 'lin', 'Emergent Charge', r'$\partial_z m_z$, arb. units', xyrange = [[-1,1],[-40,40]],nbns = 50, density = False, rescale = rescale, vmax = vmax)
        if sv != None:
            plt.savefig(sv + '.png', format = 'png')
    Nsk = np.sum(S.emb[0],axis=(1,2))
    Nsat = np.sqrt(3)/2*S.m.shape[-2]*S.m.shape[-1]*(S.Q0 / 2 / np.pi)**2
    #plt.figure()
    #plt.plot(Nsk/Nsat)
    
    print('Average Mag: ' + str(np.mean(S.m[0])))
    print('Saturation: ' + str(np.round(np.mean(Nsk)/Nsat*100)) + '%')
    
    #V = S.m.shape[1]*S.m.shape[2]*S.m.shape[3] * (10**-3 * S.dx)**3 
    
    rhoBranch = (S.NBp + S.NBm) / (S.dx * 1e4)**3 / (np.prod(S.m.shape[1:]))
    
    #np.sum(np.sign(np.clip(S.EmCh,0,None)) * np.sign(np.clip(-dzMz,0,None)) + 
    #              np.sign(np.clip(-EmCh,0,None)) * np.sign(np.clip(dzMz,0,None)))/V
    
    rhoSeg = (S.NSp + S.NSm) / (S.dx * 1e4)**3 / (np.prod(S.m.shape[1:]))
    #np.sum(np.sign(np.clip(-EmCh,0,None)) * np.sign(np.clip(-dzMz,0,None)) + 
    #              np.sign(np.clip(EmCh,0,None)) * np.sign(np.clip(dzMz,0,None)))/V
    
    print('Branching Density: ' + str(np.round(rhoBranch)) + ' um^-3')
    print('Segmenting Density: ' + str(np.round(rhoSeg)) + ' um^-3')
    
    print('Total Defect Density: ' + str(np.round(rhoBranch+rhoSeg)) + ' um^-3')
    
    #return pks[:2], EmCh, dzMz
    return np.mean(Nsk)/Nsat, np.mean(S.m[0]), rhoBranch, rhoSeg
 
   
def PkHist_Legacy(S, Q0 = 15.2, Hist = True, dx = 14.7, rescale = 1, vmax = 60, sv = None):
    
    if not hasattr(S,'rhoEm'):
        
        S.ComputeMetas()
    
    pks = np.array(np.nonzero(PkMx(S.rhoEm**2,0.02**2)))
    
    EmCh = PkAttr(pks, S.rhoEm)
    dzMz = PkAttr(pks, S.dm[0,0])

    if Hist:    
        fig = SV.MakeScatt(EmCh, dzMz, 'lin', 'Emergent Charge', r'$\partial_z m_z$, arb. units', xyrange = [[-1,1],[-40,40]],nbns = 50, density = False, rescale = rescale, vmax = vmax)
        if sv != None:
            plt.savefig(sv + '.png', format = 'png')
    Nsk = np.sum(S.emb[0],axis=(1,2))
    Nsat = np.sqrt(3)/2*Q0**2
    #plt.figure()
    #plt.plot(Nsk/Nsat)
    
    print('Average Mag: ' + str(np.mean(S.m[0])))
    print('Saturation: ' + str(np.round(np.mean(Nsk)/Nsat*100)) + '%')
    
    V = S.m.shape[1]*S.m.shape[2]*S.m.shape[3] * (10**-3 * dx)**3 
    
    rhoBranch = np.sum(np.sign(np.clip(EmCh,0,None)) * np.sign(np.clip(-dzMz,0,None)) + 
                  np.sign(np.clip(-EmCh,0,None)) * np.sign(np.clip(dzMz,0,None)))/V
    
    rhoSeg = np.sum(np.sign(np.clip(-EmCh,0,None)) * np.sign(np.clip(-dzMz,0,None)) + 
                  np.sign(np.clip(EmCh,0,None)) * np.sign(np.clip(dzMz,0,None)))/V
    
    print('Branching Density: ' + str(np.round(rhoBranch)) + ' um^-3')
    print('Segmenting Density: ' + str(np.round(rhoSeg)) + ' um^-3')
    
    print('Total Defect Density: ' + str(np.round(rhoBranch+rhoSeg)) + ' um^-3')
    
    #return pks[:2], EmCh, dzMz
    return np.mean(Nsk)/Nsat, np.mean(S.m[0]), rhoBranch, rhoSeg

def PkAttr(pks, A, nh = 2):
    
    Attr = np.full(pks.shape[1],0.)
    
    
    for m in range(pks.shape[1]):
        for i in range(-nh+pks[0,m],nh+1+pks[0,m]):
            for j in range(-nh+pks[1,m],nh+1+pks[1,m]):
                for k in range(-nh+pks[2,m],nh+1+pks[2,m]):
                    
                    Attr[m] += A[np.mod(i,A.shape[0]),np.mod(j,A.shape[1]),np.mod(k,A.shape[2])]
    
    
    return Attr


def SaveSpinDen(S, filepath):
    
    hf = h5py.File(filepath)
    hf.create_dataset('m', shape = S.m.shape, dtype = np.float32, data = S.m)
    hf.create_dataset('M0', data = S.M0)
    hf.create_dataset('H0', data = S.H0)
    hf.create_dataset('voxelaspect', data = S.voxelaspect)
    hf.create_dataset('Q0', data = S.Q0)
    hf.create_dataset('beta', shape = S.beta.shape, dtype = np.float32, data = S.beta)
    hf.create_dataset('dm', shape = S.dm.shape, dtype = np.float32, data = S.dm)
    hf.create_dataset('Fden', shape = S.Fden.shape, dtype = np.float32, data = S.Fden)
    hf.create_dataset('hHeis', shape = S.hHeis.shape, dtype = np.float32, data = S.hHeis)
    hf.create_dataset('hDM', shape = S.hDM.shape, dtype = np.float32, data = S.hDM)
    hf.create_dataset('emb', shape = S.emb.shape, dtype = np.float32, data = S.emb)
    hf.create_dataset('rhoEm', shape = S.rhoEm.shape, dtype = np.float32, data = S.rhoEm)
    hf.create_dataset('rhoM', shape = S.rhoM.shape, dtype = np.float32, data = S.rhoM)
    #hf.create_dataset('DefectType', data = S.DefectType)
    asciiList = [n.encode("ascii", "ignore") for n in S.DefectType]
    hf.create_dataset('DefectType', (len(asciiList),1),'S10', asciiList)
    hf.create_dataset('NSp', data = S.NSp)
    hf.create_dataset('NSm', data = S.NSm)
    hf.create_dataset('NBp', data = S.NBp)
    hf.create_dataset('NBm', data = S.NBm)
    hf.create_dataset('dz', data = S.dz)
    hf.create_dataset('locs', data = S.locs)
    hf.create_dataset('EmCh', data = S.EmCh)
    hf.close()

def AddMetas(filepath,M0 = 1900, H0 = 250, spct = 1,Qr = -0.74246,h = np.array([-0.75,0,0])):
    
    hf = h5py.File(filepath,'r+')
    S = SpinDensity(hf['m'][:], M0, H0, spct)
    S.initF(Qr,h)
    S.ComputeMetas()
    S.MonopoleList()
    
    hf.create_dataset('M0', data = S.M0)
    hf.create_dataset('H0', data = S.H0)
    hf.create_dataset('voxelaspect', data = S.voxelaspect)
    hf.create_dataset('Q0', data = S.Q0)
    hf.create_dataset('beta', shape = S.beta.shape, dtype = np.float32, data = S.beta)
    hf.create_dataset('dm', shape = S.dm.shape, dtype = np.float32, data = S.dm)
    hf.create_dataset('Fden', shape = S.Fden.shape, dtype = np.float32, data = S.Fden)
    hf.create_dataset('hHeis', shape = S.hHeis.shape, dtype = np.float32, data = S.hHeis)
    hf.create_dataset('hDM', shape = S.hDM.shape, dtype = np.float32, data = S.hDM)
    hf.create_dataset('emb', shape = S.emb.shape, dtype = np.float32, data = S.emb)
    hf.create_dataset('rhoEm', shape = S.rhoEm.shape, dtype = np.float32, data = S.rhoEm)
    hf.create_dataset('rhoM', shape = S.rhoM.shape, dtype = np.float32, data = S.rhoM)
    #hf.create_dataset('DefectType', data = S.DefectType)
    asciiList = [n.encode("ascii", "ignore") for n in S.DefectType]
    hf.create_dataset('DefectType', (len(asciiList),1),'S10', asciiList)
    hf.create_dataset('NSp', data = S.NSp)
    hf.create_dataset('NSm', data = S.NSm)
    hf.create_dataset('NBp', data = S.NBp)
    hf.create_dataset('NBm', data = S.NBm)
    hf.create_dataset('dz', data = S.dz)
    hf.create_dataset('locs', data = S.locs)
    hf.create_dataset('EmCh', data = S.EmCh)
    hf.create_dataset('h',h.shape,np.float32,h)
    hf.create_dataset('dx',data = S.dx)
    hf.close()
        
'''
    sig = 1/ np.maximum(1e-6,hf['wts'][:])**0.5
    
    if Phantom:
        
        S = SpinDensity(-np.roll(hf['field/array'][:].transpose((3,2,0,1)),1,axis=0), 1900, 250, 1)
        
    else:
        
        S = SpinDensity(hf['m'][:], 1900, 250, 1)
    
    zrproj = (hf['ProjTh'][:]**2 + hf['ProjPsi'][:]**2).argmin()
    
    fIm = hf['fIm'][:]
    
    ftprms, ftunc, Ft = FitRaw(hf['fIm'][:], 15.6, (64.2,64.6), sigma = sig, cent = zrproj)
    
    hf.close()
    
    Cent, Lcns, Qq = PkLcns(ftprms[None,14,:])
    
    Qr = Qq * np.pi * 2 / 128
    
    return Qr
    
    S.initF(-0.78, h)
    
    S.ComputeMetas()
    
    if Phantom:
        
        Sns, Smz, SB, SS = PkHist(S, vmax = 30, rescale = 0.5, Hist = Hist)
        
    else:
    
        Sns, Smz, SB, SS = PkHist(S, vmax = 30, Hist = Hist)
        
'''
    
    