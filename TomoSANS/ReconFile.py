# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:53:53 2022

@author: bjh3
"""

import numpy as np
from . import DataPreprocess as DP
from . import ChiSq as CS
from .Classes import *
from .ObjectiveFunctions import *
from .Minimizers import *
from .Metas import *


def NewReconStage(flnm, sv, relwts = [1,2], NXits = 100):
    
    hf = h5py.File(flnm,'r')
    M0 =  hf['M0'][()]
    H0 = hf['H0'][()]
    h = hf['h'][()]
    M = SpinDensity(hf['m'][:], M0, H0, hf['voxelaspect'][()])
    M.initF(hf['Q0'][()], h)
    X = chi2(M.m.shape, hf['fIm'][()], hf['fI0'][()], hf['ProjTh'][()], hf['ProjPsi'][()], hf['NzSample'][()], wts = hf['wts'][()])
    hf.close()
    
    
    Cost = Recon(M, ObjFun, [X,relwts], SaveProgress, sv, Nits = NXits, prg = True)
    
    AddMetas(sv, M0, H0, M.voxelaspect, M.Q0, h)

def ReconFromFile(flnm, svdir, ParamsInFile = False,
                  dpth=256, spct = [1,1,1], errct = 5.,
                  NzSample = np.array([1,2,2])*860,  
                  zr = np.full(3,None), QrMask = 9, 
                  h = -0.75*np.array([1,0,0]), mzmn = -0.4,
                  M0 = 1900, H0 = 250, wavelength = 6e-8, dx = 14.7e-7,
                  Npks = [6,2,2], Paxis = np.array(None),
                  ThetaSwitch = 45, PsiSwitch = 45, SampleDims = np.array(None),
                  wavelengthdist = ['triangular', 0.1],
                  Njits = 20, Nj0 = 10, NjReduce = 10, NE = 5,
                  Nunwt = 10, Nrewt = 1, NXits = 100, Qrsgn = -1, Fwgt = 2,
                  seedfile = None):
    '''
    

    Parameters
    ----------
    flnm : STRING
        SANS data to be reconstructed. Should be an .h5 file with appropriate keys
        see DataPreprocess.LoadRaw()
    svdir : string
        Save directory for Reconstruction
    dpth : int, optional
        number of reconstruction voxels along z. The default is 256.
    NzSample : float, optional
        hz_sample / hz_MSFR The default is 860.
    spct : float, optional
        Voxel axpect hz / hx. The default is 1.
    errct : float, optional
        Minimum error in measured intensity. The default is 5..
    zr : int, optional
        Projection index where sample is roughly aligned. The default is None.
    QrMask : float, optional
        Number of pixels (radius) of low Q mask. The default is 9.
    h : numpy array of length 3, optional
        Weight of Zeeman term for z, x, and y components. 
        The default is -0.75*np.array([1,0,0]).
    mzmn : float, optional
        Desired average magnetization (z-direciton) of the seed. The default is -0.4.
    Njits : int, optional
        Number of alternating projection iterations for creating seed. The default is 20.
    Nj0 : int, optional
        Number of alternating projactions iterations before relaxing transverse 
        magnetization constraint. The default is 10.
    NjReduce : int, optional
        How often to relax the free energy. The default is 10.
    NE : int, optional
        Number of free energy reduction iterations. The default is 5.
    M0 : float, optional
        4 Pi M_s in Gauss. The default is 1900.
    H0 : float, optional
        External field, Oe. The default is 250.
    wavelength : float, optional
        neutron wavelength in cm
    Nunwt : int, optional
        Number of conjugate gradient minimization steps for unweighted errors. 
        The default is 10.
    Nrewt : int, optional
        Number of times unweighted errors are rescales. The default is 1.
    NXits : int, optional
        Number of conjugate gradient steps for weighted errors. The default is 100.
    Qrsgn : int, optional
        Sign of Qr / D. The default is -1.

    Returns
    -------
    None.

    '''
    
    ### Retreive data ###
    #####################
    
    fIm, sig, ProjTh, ProjPsi, fI0, fI0unc = DP.LoadRaw(flnm)
    
    if ParamsInFile:
        errct, NzSample, zr, QrMask, h, mzmn, M0, H0, wavelength, dx, \
            Npks, Paxis, ThetaSwitch, PsiSwitch, SampleDims, \
            wavelengthdist = DP.LoadExpParams(flnm,
            errct, NzSample, zr, QrMask, h, mzmn, M0, H0, wavelength, dx,
            Npks, Paxis, ThetaSwitch, PsiSwitch, SampleDims, wavelengthdist)
     
    mdims = (3,) + (dpth,) + fIm.shape[1:]
    M = SpinDensity((np.random.random(mdims)-.5)*2, M0, H0, spct, dx)
    
    if np.all(SampleDims != None):
        NzSample = SampleDims / np.array(mdims[1:]) / dx
        
    sig[sig<errct] = errct # Remove zero uncertainty
    fI0unc[fI0unc<errct] = errct # Remove zero uncertainty
    
    if Paxis.all() == None:
        Paxis = (0 * ProjTh).astype(int)
        Paxis += np.sign(np.clip(np.abs(ProjTh) - ThetaSwitch,0,None)).astype(int)
        Paxis += 2 * np.sign(np.clip(np.abs(ProjPsi) - PsiSwitch,0,None)).astype(int)
    
    if seedfile == None:
    
        print('Computing seed...')    
    
        counter = 0
        Qr = 0.
        
        for ax in range(3):
        
            if np.any(Paxis == ax):
                
                print('Fitting peaks for projection axis ' + str(ax))
                
                filt = Paxis == ax
                
                ### Fit Peaks and estimate |mq| ###
                ###################################
                
                if zr[ax] == None:
                    zr[ax] = np.where(ProjTh[filt]**2 + ProjPsi[filt]**2 == np.min(ProjTh[filt]**2 + ProjPsi[filt]**2))[0][0]
                
                ftprms, ftunc, Ft = DP.FitRawv2(fIm[filt], sigma = sig[filt], cent = zr[ax], PeakNum = Npks[ax])
                Cent, Lcns, Qq = DP.PkLcns(ftprms)
                Qr += Qq * np.pi * 2 / fIm.shape[1]
                fImc, fImD, Rcrvs, Qz, tsn0 = DP.LogDecon(fIm[filt], fI0, ftprms, NzSample[ax], 
                                                          ProjTh[filt], ProjPsi[filt], cdx = zr[ax],
                                                          Paxis = ax)
                
                if counter == 0:
                    
                    tsn = tsn0
                    
                else:
                    
                    tsn = np.maximum(tsn,tsn0)
                    
                counter += 1
                
        
        ### Compute and save D/J ###
        ############################
        Qr *= 1/ counter
        #Qr = Qq * np.pi * 2 / M.m.shape[3]
        M.initF(Qrsgn * Qr, h)
        
        ### Setup lowQ mask and ChiSq ###
        #################################
        
        qx = np.linspace(0,M.m.shape[2]-1,M.m.shape[2])
        qy = np.linspace(0,M.m.shape[3]-1,M.m.shape[3])
        rr = (qx[:,None] - Cent[0])**2 + (qy[None,:]-Cent[1])**2
        R0Msk = np.sign(np.clip(rr - QrMask**2,0,None))
        
        wts0 = R0Msk / sig**2
        X = chi2(M.m.shape, fIm, fI0, ProjTh, ProjPsi, NzSample, wts0, 
                 wavelengthdist, wavelength, dx, Paxis, spct = spct)
        
        ### Renormalize |mq| = tsn ###
        ##############################
        
        tsn *= (1-1*mzmn**2)/np.sum(tsn)
        tsn[0,0,0] = 1*np.abs(mzmn)**2
        
        ### Create Seed with Alternating Projections ###
        ################################################
        
        
        
        erj = AltProj(M, tsn, Njits, Nj0, NjReduce, NE)
        
    else:
        
        hf = h5py.File(seedfile)
        M.initF(hf['Q0'][()], h)
        hf.close()
        
        qx = np.linspace(0,M.m.shape[2]-1,M.m.shape[2])
        qy = np.linspace(0,M.m.shape[3]-1,M.m.shape[3])
        rr = (qx[:,None] - Cent[0])**2 + (qy[None,:]-Cent[1])**2
        R0Msk = np.sign(np.clip(rr - QrMask**2,0,None))
        
        wts0 = R0Msk / sig**2
        X = chi2(M.m.shape, fIm, fI0, ProjTh, ProjPsi, NzSample, wts0, 
                 wavelengthdist, wavelength, dx, Paxis, spct = spct)
            
    X.ComputeResids(M) #residuals of seed
    
    for j in range(Nrewt):
        '''
        Reconstruct with equal weights for Nunwt iterations Nrewt times
        save outputs
        '''
        print('Running unweighted reconstruction ' + str(j+1) + ' of ' + str(Nrewt) + '...')
        print('You can monitor progress by looking at the following file:')
        
        
        fl = svdir + 'recon' + str(j) + '.h5'
        print(fl)
        ch0 = np.sum((X.fIm - X.Ikt)**2 * R0Msk)
        chisq = np.sum((X.fIm - X.Ikt)**2 * wts0)
        X.weights = 0. * X.weights + R0Msk * chisq / ch0
    
        Cost0 = Recon(M, ObjFun, [X,[1,Fwgt]], SaveProgress, fl, Nits = Nunwt, prg = True)
    
    ### Final Reconstruction with Experimental Weights ###
    ######################################################
    
    j += 1
    fl = svdir + 'recon' + str(j) + '.h5'
    X.weights = 0. + wts0
    
    print('Running the weighted reconstruction...')
    print('You can monitor progress by looking at the following file:')
    print(fl)
    Cost1 = Recon(M, ObjFun, [X,[1,Fwgt]], SaveProgress, fl, Nits = NXits, prg = True)
    
    
    ### Compute and Save Reconstruction Meta Data ###
    #################################################
    print('Reconstructions Complete!')
    print('Computing MP/AMP locations and saving to reconstruction file...')
    AddMetas(fl, M0, H0, M.voxelaspect, M.Q0, h)
    
    
        
def AltProj(M, tsn, Njits,Nj0,NjReduce,NE):
    q = CS.MakeQ(M.m.shape, spct = M.voxelaspect)
    erj = np.full(Njits,0.)
    
    for j in range(Njits):
    
        fmg = np.fft.ifftn(M.m,axes=(1,2,3))
        erj[j] = np.sum((np.abs(tsn)-np.sum(np.abs(fmg)**2,axis=0))**2 )#np.sign(np.clip(rs-8**2,0,1)))
        
        fmg *= np.sqrt(np.abs(tsn)/(1e-12 + np.sum(np.abs(fmg)**2,axis=0)))
        
        if j < Nj0:
            fmg[1] = -1.j * q[2] * fmg[0]
            fmg[2] = 1.j * q[1] * fmg[0]
            M.m = np.real(np.fft.fftn(fmg,axes=(1,2,3)))
            M.m = np.clip(M.m,-1,1)
            TH = np.arctan2(np.sqrt(M.m[1]**2 + M.m[2]**2), M.m[0])
            PH = np.arctan2(M.m[2],M.m[1])
            
            M.m[1] = np.sin(TH) * np.cos(PH)
            M.m[2] = np.sin(TH) * np.sin(PH)
            M.m[0] = np.cos(TH)
            
        else:
            
            M.m = np.real(np.fft.fftn(fmg,axes=(1,2,3)))
            M.m = np.clip(M.m,-1,1)
                
            TH = np.arctan2(np.sqrt(M.m[1]**2 + M.m[2]**2),M.m[0])
            PH = np.arctan2(M.m[2],M.m[1])
            
            M.m[1] = np.sin(TH) * np.cos(PH)
            M.m[2] = np.sin(TH) * np.sin(PH)
            M.m[0] = np.cos(TH) 
                
            if j % NjReduce == 0:
                
                print('Running a Free energy reduction...')
                Cost = Recon(M,EMin, Nits = NE, prg = True)
                
          
        if j > 4 and np.mean(M.m[0]) > 0:
            M.m *= -1.
            
    return erj
    
    
