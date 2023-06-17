# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:25:07 2021

@author: bjh3
"""

from scipy.optimize import curve_fit
import numpy as np
import h5py
from scipy import ndimage
from tqdm import tqdm
from . import DeconvolutionToolbox as DT
import matplotlib.pyplot as plt

def LoadRawNIST(flnm, st = 0, en = None):

    #st = 2
    #en = 27
    hf = h5py.File(flnm,'r')
    SANS = hf['SANS'][st:en]
    SANSUnc = hf['SANSUnc'][st:en]
    Flare = hf['Flare'][:]
    FlareUnc = hf['FlareUnc'][:]
    EmptBm = hf['EmptBm'][:]
    EmptBmUnc = hf['EmptBmUnc'][:]
    Proj = hf['Proj'][st:en]
    hf.close()

    SigT =   -np.log(0.185) # (3.4/6.35**3) *0.1 *((8 * 37 + 8 * 1.11 + 4 * 3 )* 6/1.8 + 8 * 4.8 + 8 * 0.077 + 4 * 0.4 ) #
    ArRat = ((np.pi *(9.5/2)**2 ) /(3.3 *3))**-1 / 1.
    
    fI0 = EmptBm * ArRat * np.exp(-SigT) 
    
    fIm = np.clip(SANS-Flare[None,:],0,None)
    
    sig = (SANSUnc**2 + FlareUnc[None,:]**2)**0.5
    
    return fIm, sig, Proj*0., Proj, fI0

def LoadRaw(flnm, st = 0, en = None):
    '''
    

    Parameters
    ----------
    flnm : STRING
        .h5 filename containing SANS data
    st : int, optional
        Starting index to import. The default is 0.
    en : int, optional
        Ending index for import. The default is None.

    Returns
    -------
    fIm : Nproj x NQx x NQy numpy array
        Measured intensity
    sig : Nproj x NQx x NQy numpy array
        measurement uncertainties
    ProjTh : Nproj numpy array
        Projection angles about horizontal axis
    ProjPsi : Nproj numpy array
        Projection angles about vertical axis
        Compound rotations currently NOT supported
    fI0 : NQx x NQy numpy array
        Transmission data
        Transmission data for each projection currently NOT supported
    EmptBmUnc : NQx x NQy numpy array
        fI0 uncertainty

    '''

    #st = 2
    #en = 27
    hf = h5py.File(flnm,'r')
    fIm = hf['SANS'][st:en]
    sig = hf['SANSUnc'][st:en]
    #Flare = hf['Flare'][:]
    #FlareUnc = hf['FlareUnc'][:]
    fI0 = hf['EmptBm'][:]
    EmptBmUnc = hf['EmptBmUnc'][:]
    ProjTh = hf['ProjTh'][st:en]
    ProjPsi = hf['ProjPsi'][st:en]
    
    #Qx = hf['Qx'][()]
    #Qy = hf['Qy'][()]
    
    hf.close()

    #SigT =   -np.log(0.185) # (3.4/6.35**3) *0.1 *((8 * 37 + 8 * 1.11 + 4 * 3 )* 6/1.8 + 8 * 4.8 + 8 * 0.077 + 4 * 0.4 ) #
    #ArRat = ((np.pi *(9.5/2)**2 ) /(3.3 *3))**-1 / 1.
    
    #fI0 = EmptBm #* ArRat * np.exp(-SigT) 
    
    #fIm = np.clip(SANS-Flare[None,:],0,None)
    
    #sig = (SANSUnc**2 + FlareUnc[None,:]**2)**0.5
    
    return fIm, sig, ProjTh, ProjPsi, fI0, EmptBmUnc #, Qx, Qy

def LoadExpParams(flnm, errct, NzSample, zr, QrMask, h, mzmn,
                  M0, H0, wavelength, dx,
                  Npks, Paxis, ThetaSwitch, PsiSwitch, SampleDims, wavelengthdist):
    
    hf = h5py.File(flnm,'r')
    
    if 'errct' in hf:
        errct = hf['errct'][()]
    
    if 'NzSample' in hf:
        NzSample = hf['NzSample'][()]
    
    if 'zr' in hf:
        zr = hf['zr'][()]
    
    if 'QrMask' in hf: 
        QrMask = hf['QrMask'][()]
        
    if 'h' in hf:
        h = hf['h'][()]
    
    if 'mzmn' in hf:
        mzmn = hf['mzmn'][()]
        
    if 'M0' in hf:
        M0 = hf['M0'][()]
        
    if 'H0' in hf:
        H0 = hf['H0'][()]
        
    if 'wavelength' in hf:
        wavelength = hf['wavelength'][()]
        
    if 'dx' in hf:
        dx = hf['dx'][()]
                      
    if 'Npks' in hf:
        Npks = hf['Npks'][()]
        
    if 'Paxis' in hf:
        Paxis = hf['Paxis'][()] 
    
    if 'ThetaSwitch' in hf:
        ThetaSwitch = hf['ThetaSwitch'][()]
        
    if 'PsiSwitch' in hf:
        PsiSwitch = hf['PsiSwitch'][()] 
        
    if 'SampleDims' in hf:
        SampleDims = hf['SampleDims']
        
    if 'wavelengthdist' in hf:
        wavelengthdist = [hf['wavelengthdist'][()], hf['wavelength_spread'][()]]
        
    return errct, NzSample, zr, QrMask, h, mzmn, M0, H0, wavelength, dx, \
                      Npks, Paxis, ThetaSwitch, PsiSwitch, SampleDims, wavelengthdist

def LoadRawSim(flnm, st = 0, en = None):

    #st = 2
    #en = 27
    hf = h5py.File(flnm,'r')
    fIm = hf['SANS'][st:en]
    sig = hf['SANSUnc'][st:en]
    #Flare = hf['Flare'][:]
    #FlareUnc = hf['FlareUnc'][:]
    fI0 = hf['EmptBm'][:]
    EmptBmUnc = hf['EmptBmUnc'][:]
    ProjTh = hf['ProjTh'][st:en]
    ProjPsi = hf['ProjPsi'][st:en]
    hf.close()

    #SigT =   -np.log(0.185) # (3.4/6.35**3) *0.1 *((8 * 37 + 8 * 1.11 + 4 * 3 )* 6/1.8 + 8 * 4.8 + 8 * 0.077 + 4 * 0.4 ) #
    #ArRat = ((np.pi *(9.5/2)**2 ) /(3.3 *3))**-1 / 1.
    
    #fI0 = EmptBm #* ArRat * np.exp(-SigT) 
    
    #fIm = np.clip(SANS-Flare[None,:],0,None)
    
    #sig = (SANSUnc**2 + FlareUnc[None,:]**2)**0.5
    
    return fIm, sig, ProjTh, ProjPsi, fI0

def IQxy(Qxy,A,Q0,Gxy,Gr,sg):
    
    #Qz = Q[2]
    #Qy = Q[1]
    #Qx = Q[0]
    
    rtn = np.full(Qxy[0].shape,0.)
    
    Qc = np.full(2,0.)
    Qc = np.mean(Q0,axis = 0) #*Q0.shape[1]/2
    Qr = np.mean(((Q0[:,0]-Qc[0])**2 + (Q0[:,1]-Qc[1])**2)**0.5)
    
    Qxyr = ((Qxy[0]-Qc[0])**2 + (Qxy[1]-Qc[1])**2 )**0.5
    
    for j in range(Q0.shape[0]): 
       
        #Q0c = (Q0[j,0]-Qc[0]) / Qr * Qxyr
        #Q0s = (Q0[j,1]-Qc[1]) / Qr * Qxyr
        rtn += ndimage.gaussian_filter(A[j] / ((Qxy[0]-Q0[j,0])**2 / Gxy[j]**2 + (Qxy[1]-Q0[j,1])**2 / Gxy[j]**2 + 1 )\
            / ((((Qxy[0]-Qc[0])**2 + (Qxy[1]-Qc[1])**2 )**0.5-Qr)**2 / Gr**2 + 1),sg)
            
        #rtn += A[j] / ((Qxy[0]-Qc[0]-Q0c)**2 / Gxy[j]**2 + (Qxy[1]-Qc[1]-Q0s)**2 / Gxy[j]**2  +  \
        #    (((Qxy[0]-Qc[0])**2 + (Qxy[1]-Qc[1])**2 )**0.5-Qr)**2 * Gr**2 + 1) \
        #    /((((Qxy[0]-Qc[0])**2 + (Qxy[1]-Qc[1])**2 )**0.5-Qr)**2 * Gr**2 / 100 + 1)
        #    Qxy[0]**2 / Gz**2 + 1)
        
    return rtn



def IQxyzFt(Qxy,*args):
    
    Na = len(args[0][:])
    
    Npks = int((Na-2)/4)
    
    A = np.array(args[0][:Npks])
    Q0 = np.array(args[0][Npks:3*Npks]).reshape((Npks,2))
    Gxy = args[0][3*Npks:4*Npks]
    Gr = args[0][4*Npks]
    Gz = args[0][4*Npks+1]
    
    return IQxyz(Qxy,A,Q0,Gxy,Gr,Gz)

def FixfI0(fI0,x0,y0,sg):
    
    fx = int(np.floor(x0))
    fy = int(np.floor(y0))
    fI00 = 0. * fI0
    fI00[fx,fy] = (1 + fx - x0)*(1 + fy - y0) 
    fI00[fx+1,fy] = (- fx + x0)*(1 + fy - y0) 
    fI00[fx,fy+1] = (1 + fx - x0)*(- fy + y0)
    fI00[fx+1,fy+1] = (- fx + x0)*(- fy + y0)
    
    fI00 *= np.sum(fI0)/np.sum(fI00)
    
    fI00 = ndimage.gaussian_filter(fI00,sg)
    
    return fI00

def IQxyz(Qxy,A,Q0,Gxy,Gr,Gz):
    
    #Qz = Q[2]
    #Qy = Q[1]
    #Qx = Q[0]
    
    rtn = np.full(Qxy[0].shape,0.)
    
    Qc = np.full(2,0.)
    Qc = np.mean(Q0,axis = 0) #*Q0.shape[1]/2
    Qr = np.mean(((Q0[:,0]-Qc[0])**2 + (Q0[:,1]-Qc[1])**2)**0.5)
    
    
    Qxyr = ((Qxy[1]-Qc[0])**2 + (Qxy[2]-Qc[1])**2 )**0.5
    #Qxyc = (Qxy[1]-Qc[0])/ (1e-8 + (Qxy[1]-Qc[0])**2 + (Qxy[2]-Qc[1])**2 )**0.5
    #Qxys = (Qxy[2]-Qc[1])/ (1e-8 + (Qxy[1]-Qc[0])**2 + (Qxy[2]-Qc[1])**2 )**0.5
    
    #print(Qc)
    #print(Qr)
    for j in range(Q0.shape[0]): 
        #Q0c = (Q0[j,0]-Qc[0]) / Qr * Qxyr
        #Q0s = (Q0[j,1]-Qc[1]) / Qr * Qxyr
        Qzr = (Qr**2 + Qxy[0]**2)**0.5
        Q0c = Q0[j,0] - Qc[0]#/ Qzr * Qr
        Q0s = Q0[j,1] - Qc[1]#/ Qzr * Qr
        #print(Q0c)
        #print(Q0s)
        #rtn += A[j] / ((Qxyc-Q0c)**2 / Gxy[j]**2 + (Qxys-Q0s)**2 / Gxy[j]**2  + 1) \
        #    /((((Qxy[1]-Qc[0])**2 + (Qxy[2]-Qc[1])**2 + Qxy[0]**2 )**0.5-Qr)**2 / Gr**2 + 1) \
        #   /( Qxy[0]**2 / Gz**2 + 1)
        
        rtn += A[j] / ((Qxy[1]-Q0c)**2 / Gxy[j]**2 + (Qxy[2]-Q0s)**2 / Gxy[j]**2  + 1) \
           /( (((Qxy[1]-0.*Qc[0])**2 + (Qxy[2]-0.*Qc[1])**2 + Qxy[0]**2 )**0.5-Qr)**2 / Gr**2 + 1) \
           /( Qxy[0]**2 / Gz**2 + 1)
                
        #Qr = ((Q0[j,0]-Qc[0])**2 + (Q0[j,1]-Qc[1])**2)**0.5
        #rtn += A[j] / ((Qx-Q0[j,0]+Qc[0])**2 / Gxy[j]**2 + (Qy-Q0[j,1]+Qc[1])**2 / Gxy[j]**2 + Qz**2 / Gz**2 + 1) / ((((Qx)**2 + (Qy)**2 + Qz**2)**0.5-Qr)**2 / Gr**2 + 1)
    #return Q0c
    
    '''
    for j in range(Q0.shape[0]): 
       
        #Q0c = (Q0[j,0]-Qc[0]) / Qr * Qxyr
        #Q0s = (Q0[j,1]-Qc[1]) / Qr * Qxyr
        rtn += A[j] / ((Qxy[0]-Q0[j,0])**2 / Gxy[1]**2 + (Qxy[1]-Q0[j,1])**2 / Gxy[j]**2 + 1 )\
            / ((((Qxy[0]-Qc[0])**2 + (Qxy[1]-Qc[1])**2 )**0.5-Qr)**2 / Gr**2 + 1)
    '''
    return rtn

def IQxyFt(Qxy,*args):
    
    Na = len(args[0][:])
    
    Npks = int((Na-2)/4)
    
    A = np.array(args[0][:Npks])
    Q0 = np.array(args[0][Npks:3*Npks]).reshape((Npks,2))
    Gxy = args[0][3*Npks:4*Npks]
    Gr = args[0][4*Npks]
    sg = args[0][4*Npks + 1]
    
    
    return IQxy(Qxy,A,Q0,Gxy,Gr,sg)

def UnFlatParams(*args):
    
    
    
    Na = args[0].shape[1]
    
    Npks = int((Na-2)/4)
    
    A = args[0][:,:Npks]
    Q0 = args[0][:,Npks:3*Npks].reshape((args[0].shape[0],Npks,2))
    Gxy = args[0][:,3*Npks:4*Npks]
    Gr = args[0][:,4*Npks]
    sg = args[0][:,4*Npks + 1]
    
    
    return A, Q0, Gxy, Gr, sg


def PkLcns(args):
    
    A, Q0, Gxy, Gr, sg = UnFlatParams(args + 0.)
    
    #Center = np.sum(A[:,:,None] * Gxy[:,:,None]**2 * Q0, axis = (0,1)) / np.sum(A * Gxy**2)
    Center = np.mean(Q0,axis=(0,1))
    Lcns = np.sum(A[:,:,None] * Gxy[:,:,None]**2 * Q0,axis = 0) / np.sum(A * Gxy**2, axis=0)[:,None]
    
    Q0 = (np.sum((Lcns - Center)**2)/A.shape[1])**0.5
    
    return Center, Lcns, Q0

def Recenter(fI0,Cent,sg):
    
    #Cent0 = np.array(ndimage.center_of_mass(fI0))
    fI00 = 0. * fI0
    fI00[0,0] = 1.
    fI00 = ndimage.gaussian_filter(fI00,1,mode = 'wrap')
    fx = int(np.floor(Cent[0]))
    fy = int(np.floor(Cent[1]))
    x0 = Cent[0]
    y0 = Cent[1]
    fI00 = ((1 + fx - x0)*(1 + fy - y0) * np.roll(fI00,(fx,fy),axis = (0,1))
           + (- fx + x0)*(1 + fy - y0) * np.roll(fI00,(1+fx,fy),axis = (0,1))
           + (1 + fx - x0)*(- fy + y0)* np.roll(fI00,(fx,1+fy),axis = (0,1))
           + (- fx + x0)*(- fy + y0)* np.roll(fI00,(1+fx,1+fy),axis = (0,1)))
    fI00 *= np.sum(fI0)
    #fI00 = ndimage.gaussian_filter(np.clip(ndimage.shift(fI00,Cent-Cent.astype(int)),0,None) * np.sum(fI0),sg)
    
    return fI00

def LogDecon(fIm, fI0, ftprms, N, ProjTh, ProjPsi,
             dims = (256,128,128), cdx = 7, Nits = 20, reg = 1e-12, sprd = 3, 
             qr0 = 5, Paxis = 0, spct = [1,1,1]):
    
    Cent, Lcns, Q0 = PkLcns(ftprms[None,cdx,:])
    
    fImc = np.clip(ndimage.shift(fIm, (0, -Cent[0], -Cent[1]), mode = 'grid-wrap'),0,None)
    if len(fI0.shape)==2:
        fI0c = np.clip(ndimage.shift(fI0, -Cent, mode = 'grid-wrap'),0,None)
        
        fImc += fI0c / np.sum(fI0c) * (np.sum(fI0c) - np.sum(fImc,axis=(1,2)))[:,None,None]
        
        fImc *= 1/ np.sum(fI0c)
        fI0c *= 1/ np.sum(fI0c)
    else:
        fI0c = np.clip(ndimage.shift(fI0, (0, -Cent[0], -Cent[1]), mode = 'grid-wrap'),0,None)
        
        fImc += fI0c *(1 / np.sum(fI0c,axis=(1,2)) * (np.sum(fI0c,axis=(1,2)) - np.sum(fImc,axis=(1,2))))[:,None,None]
        
        fImc *= 1/ np.sum(fI0c,axis=(1,2))[:,None,None]
        fI0c *= 1/ np.sum(fI0c,axis=(1,2))[:,None,None]
    
    
    
    #return fImc, fI0c
    
    if N < 400:
        fImc = DT.DeconSignalSng(fImc, fI0c, N = N, Nits = Nits, reg = reg)
    else:
        fImc = DT.DeconSignalSng(fImc, fI0c, N = N/20., N2 = 20, Nits = Nits, reg = reg)    
    
    #return fImc
    
    fImD = ndimage.shift(fImc, (0, Cent[0], Cent[1]), mode = 'grid-wrap')
    
    Rcrvs, Qz = np.full((2,fImc.shape[0],Lcns.shape[0]),0.)
    
    WeightSum = 0
    G = 0.
    
    if Paxis == 0:
        
        theta = 0. + ProjTh
        psi = 0. + ProjPsi
        
    elif Paxis == 1:
        
        theta = np.sign(ProjTh) * 90 - ProjTh
        psi = 0. + ProjPsi
        
    else:
        
        theta = 0. + ProjTh
        psi = np.sign(ProjPsi) * 90 - ProjPsi
    
    for j in range(Lcns.shape[0]):
        
        lx = int(Lcns[j,0])
        ly = int(Lcns[j,1])
        
        Rcrvs[:,j] = np.sum(fImD[:,lx-sprd:lx+sprd,ly-sprd:ly+sprd],axis=(1,2))
        
        Qz[:,j] = (Lcns[j,1]-Cent[1])*np.tan(psi/180*np.pi) \
            + (Lcns[j,0]-Cent[0])*np.tan(theta/180*np.pi)
    
        #return Qz, Rcrvs
        if np.max(Qz[:,j])-np.min(Qz[:,j]) > 0.1:
            ftob = curve_fit(FitLor, Qz[:,j], Rcrvs[:,j], p0 = (1e-4,0,0.5))
            G += ftob[0][-1] / ftob[1][-1,-1]
            WeightSum += 1 / ftob[1][-1,-1]
    
    tsn = np.full(dims,0.)
    
    if WeightSum == 0:
        
        return fImc, fImD, Rcrvs, Qz, tsn
    
    G *= 1/WeightSum
    qz = np.roll(np.linspace(0,dims[0]-1,dims[0]) - int(dims[0]/2),int(dims[0]/2))
    qx = np.roll(np.linspace(0,dims[1]-1,dims[1]) - int(dims[1]/2),int(dims[1]/2))
    qy = np.roll(np.linspace(0,dims[2]-1,dims[2]) - int(dims[2]/2),int(dims[2]/2))
    r0msk = np.sign(np.clip(qx[:,None]**2 + qy[None,:]**2 - qr0**2,0,None))
    
    shft = np.array([dims[1]/2,dims[2]/2])
    tsn[0] = fImc[cdx] * r0msk
    if Paxis == 2:
        zr = ndimage.shift((fImc[cdx] * r0msk).transpose(), shft, mode = 'grid-wrap') 
    else:
        zr = ndimage.shift(fImc[cdx] * r0msk, shft, mode = 'grid-wrap') 
    
    zoom = (1,1)
    j = 1
    while np.all(np.array(zoom) > 0):
        
        if Paxis == 0:
        
            zoom = (np.clip(Q0**2-(dims[1]/dims[0]*j)**2,0,None)**0.5/Q0 *dims[1] / zr.shape[0],
                    np.clip(Q0**2-(dims[2]/dims[0]*j)**2,0,None)**0.5/Q0 *dims[2] / zr.shape[1]) 
            zm = ndimage.zoom(zr, zoom) * zr.shape[0] / dims[1] * zr.shape[1] / dims[2]
            #print(zoom)
            pd0n = int(np.floor((dims[1]-zm.shape[0])/2))
            pd0p = int(np.ceil((dims[1]-zm.shape[0])/2))
            pd1n = int(np.floor((dims[2]-zm.shape[1])/2))
            pd1p = int(np.ceil((dims[2]-zm.shape[1])/2))
            zm = np.pad(zm, ((pd0n,pd0p),(pd1n,pd1p)))
            offst = np.array([(pd0p - pd0n-0.)/2,(pd1p - pd1n - 0.)/2])
            #com = ndimage.measurements.center_of_mass(zm)
            #print(com)
            zm = ndimage.shift(zm, -shft+offst, mode = 'grid-wrap')
            tsn[j] = tsn[-j] = np.clip(zm,0,None) / ((np.sqrt(dims[1]*dims[2])/dims[0]*j)**2 / G**2 + 1)
            j+=1
            
        elif Paxis == 1:
            
            zoom = (np.clip(Q0**2-(dims[0]/dims[1]*j)**2,0,None)**0.5/Q0 * dims[0] / zr.shape[0],
                    np.clip(Q0**2-(dims[2]/dims[1]*j)**2,0,None)**0.5/Q0 * dims[2] / zr.shape[1])
            zm = ndimage.zoom(zr, (zoom)) * zr.shape[0] / dims[0] * zr.shape[1] / dims[2]
            #print(zoom)
            pd0n = int(np.floor((dims[0]-zm.shape[0])/2))
            pd0p = int(np.ceil((dims[0]-zm.shape[0])/2))
            pd1n = int(np.floor((dims[2]-zm.shape[1])/2))
            pd1p = int(np.ceil((dims[2]-zm.shape[1])/2))
            zm = np.pad(zm, ((pd0n,pd0p),(pd1n,pd1p)))
            offst = np.array([(pd0p - pd0n-0.)/2,(pd1p - pd1n - 0.)/2])
            #com = ndimage.measurements.center_of_mass(zm)
            #print(com)
            zm = ndimage.shift(zm, -shft+offst, mode = 'grid-wrap')
            tsn[:,j] = tsn[:,-j] = np.clip(zm,0,None) / ((np.sqrt(dims[0]*dims[2])/dims[1]*j)**2 / G**2 + 1)
            j+=1
            
        else:
            
            zoom = (np.clip(Q0**2-(dims[0]/dims[2]*j)**2,0,None)**0.5/Q0 *dims[0] / zr.shape[1],
                    np.clip(Q0**2-(dims[1]/dims[2]*j)**2,0,None)**0.5/Q0 *dims[1] / zr.shape[0])
            zm = ndimage.zoom(zr, zoom)  * zr.shape[1] / dims[0] * zr.shape[0] / dims[1]
            #print(zoom)
            pd0n = int(np.floor((dims[0]-zm.shape[0])/2))
            pd0p = int(np.ceil((dims[0]-zm.shape[0])/2))
            pd1n = int(np.floor((dims[1]-zm.shape[1])/2))
            pd1p = int(np.ceil((dims[1]-zm.shape[1])/2))
            zm = np.pad(zm, ((pd0n,pd0p),(pd1n,pd1p)))
            offst = np.array([(pd0p - pd0n-0.)/2,(pd1p - pd1n - 0.)/2])
            #com = ndimage.measurements.center_of_mass(zm)
            #print(com)
            zm = ndimage.shift(zm, -shft+offst, mode = 'grid-wrap')
            tsn[:,:,j] = tsn[:,:,-j] = np.clip(zm,0,None) / ((np.sqrt(dims[0]*dims[1])/dims[2]*j)**2 / G**2 + 1)
            j+=1
    
    return fImc, fImD, Rcrvs, Qz, tsn
    

def LogDeconPaxis(fIm, fI0, ftprms, N, ProjTh, ProjPsi, 
             dims = (256,128,128), cdx = 7, Nits = 20, reg = 1e-12, 
             sprd = 3, qr0 = 5, Paxis = 0):
    
    Cent, Lcns, Q0 = PkLcns(ftprms[None,cdx,:])
    
    fImc = np.clip(ndimage.shift(fIm, (0, -Cent[0], -Cent[1]), mode = 'grid-wrap'),0,None)
    fI0c = np.clip(ndimage.shift(fI0, -Cent, mode = 'grid-wrap'),0,None)
    
    fImc += fI0c / np.sum(fI0c) * (np.sum(fI0c) - np.sum(fImc,axis=(1,2)))[:,None,None]
    
    fImc *= 1/ np.sum(fI0c)
    fI0c *= 1/ np.sum(fI0c)
    
    #return fImc, fI0c
    
    if N < 400:
        fImc = DT.DeconSignalSng(fImc, fI0c, N = N, Nits = Nits, reg = reg)
    else:
        fImc = DT.DeconSignalSng(fImc, fI0c, N = N/20., N2 = 20, Nits = Nits, reg = reg)    
    
    #return fImc
    
    fImD = ndimage.shift(fImc, (0, Cent[0], Cent[1]), mode = 'grid-wrap')
    
    Rcrvs, Qz = np.full((2,fImc.shape[0],Lcns.shape[0]),0.)
    
    WeightSum = 0
    G = 0.
    
    for j in range(Lcns.shape[0]):
        
        lx = int(Lcns[j,0])
        ly = int(Lcns[j,1])
        
        Rcrvs[:,j] = np.sum(fImD[:,lx-sprd:lx+sprd,ly-sprd:ly+sprd],axis=(1,2))
        
        Qz[:,j] = (Lcns[j,1]-Cent[1])*np.tan(ProjPsi/180*np.pi) \
            + (Lcns[j,0]-Cent[0])*np.tan(ProjTh/180*np.pi)
    
        #return Qz, Rcrvs
        if np.max(Qz[:,j])-np.min(Qz[:,j]) > 0.1:
            ftob = curve_fit(FitLor, Qz[:,j], Rcrvs[:,j], p0 = (1e-4,0,0.5))
            G += ftob[0][-1] / ftob[1][-1,-1]
            WeightSum += 1 / ftob[1][-1,-1]
            
    G *= 1/WeightSum
    tsn = np.full(dims,0.)
    qz = np.roll(np.linspace(0,dims[1]-1,dims[1]) - int(dims[1]/2),int(dims[1]/2))
    qx = np.roll(np.linspace(0,dims[1]-1,dims[1]) - int(dims[1]/2),int(dims[1]/2))
    qy = np.roll(np.linspace(0,dims[2]-1,dims[2]) - int(dims[2]/2),int(dims[2]/2))
    r0msk = np.sign(np.clip(qx[:,None]**2 + qy[None,:]**2 - qr0**2,0,None))
    
    shft = np.array([dims[1]/2,dims[2]/2])
    tsn[0] = fImc[cdx] * r0msk
    zr = ndimage.shift(fImc[cdx] * r0msk, shft, mode = 'grid-wrap') 
    
    zoom = 1
    j = 1
    while zoom > 0:
        
        if Paxis == 0:
        
            zoom = (np.clip(Q0**2-(dims[1]/dims[0]*j)**2,0,None)**0.5/Q0,
                    np.clip(Q0**2-(dims[2]/dims[0]*j)**2,0,None)**0.5/Q0)
    
            zm = ndimage.zoom(zr, zoom)
            #print(zoom)
            pd0n = int(np.floor((dims[1]-zm.shape[0])/2))
            pd0p = int(np.ceil((dims[1]-zm.shape[0])/2))
            pd1n = int(np.floor((dims[2]-zm.shape[1])/2))
            pd1p = int(np.ceil((dims[2]-zm.shape[1])/2))
            zm = np.pad(zm, ((pd0n,pd0p),(pd1n,pd1p)))
            offst = np.array([(pd0p - pd0n-0.)/2,(pd1p - pd1n - 0.)/2])
            #com = ndimage.measurements.center_of_mass(zm)
            #print(com)
            zm = ndimage.shift(zm, -shft+offst, mode = 'grid-wrap')
            tsn[j] = tsn[-j] = np.clip(zm,0,None) / ((np.sqrt(dims[1]*dims[2])/dims[0]*j)**2 / G**2 + 1)
            j+=1
            
        elif Paxis == 1:
        
            zoom = (np.clip(Q0**2-(dims[0]/dims[1]*j)**2,0,None)**0.5/Q0,
                    np.clip(Q0**2-(dims[2]/dims[1]*j)**2,0,None)**0.5/Q0)
    
            zm = ndimage.zoom(zr, zoom)
            #print(zoom)
            pd0n = int(np.floor((dims[0]-zm.shape[0])/2))
            pd0p = int(np.ceil((dims[0]-zm.shape[0])/2))
            pd1n = int(np.floor((dims[2]-zm.shape[1])/2))
            pd1p = int(np.ceil((dims[2]-zm.shape[1])/2))
            zm = np.pad(zm, ((pd0n,pd0p),(pd1n,pd1p)))
            offst = np.array([(pd0p - pd0n-0.)/2,(pd1p - pd1n - 0.)/2])
            #com = ndimage.measurements.center_of_mass(zm)
            #print(com)
            zm = ndimage.shift(zm, -shft+offst, mode = 'grid-wrap')
            tsn[:,j] = tsn[:,-j] = np.clip(zm,0,None) / ((np.sqrt(dims[0]*dims[2])/dims[1]*j)**2 / G**2 + 1)
            j+=1
    
        else:
        
            zoom = (np.clip(Q0**2-(dims[0]/dims[2]*j)**2,0,None)**0.5/Q0,
                    np.clip(Q0**2-(dims[1]/dims[2]*j)**2,0,None)**0.5/Q0)
    
            zm = ndimage.zoom(zr, zoom)
            #print(zoom)
            pd0n = int(np.floor((dims[0]-zm.shape[0])/2))
            pd0p = int(np.ceil((dims[0]-zm.shape[0])/2))
            pd1n = int(np.floor((dims[1]-zm.shape[1])/2))
            pd1p = int(np.ceil((dims[1]-zm.shape[1])/2))
            zm = np.pad(zm, ((pd0n,pd0p),(pd1n,pd1p)))
            offst = np.array([(pd0p - pd0n-0.)/2,(pd1p - pd1n - 0.)/2])
            #com = ndimage.measurements.center_of_mass(zm)
            #print(com)
            zm = ndimage.shift(zm, -shft+offst, mode = 'grid-wrap')
            tsn[:,:,j] = tsn[:,:,-j] = np.clip(zm,0,None) / ((np.sqrt(dims[0]*dims[1])/dims[2]*j)**2 / G**2 + 1)
            j+=1    
    
    return fImc, fImD, Rcrvs, Qz, tsn


def FitLors(ProjTh, ProjPsi, args, uncs, dims = (256,128,128), PlotResults = False):
    
    A, Q0, Gxy, Gr, sg = UnFlatParams(args + 0.)
    Ae, Q0e, Gxye, Gre, sge = UnFlatParams(0. + uncs**0.5)
    
    sigma = np.sqrt(Gxy**4 * Ae**2 + (2*Gxy*A)**2 * Gxye**2)
    
    
    
    A *= Gxy**2
    Qz = (Q0[:,:,1]-np.mean(Q0[:,:,1]))*np.tan(ProjPsi[:,None]/180*np.pi) \
        + (Q0[:,:,0]-np.mean(Q0[:,:,0]))*np.tan(ProjTh[:,None]/180*np.pi)
    
    curvelist = []
    #plt.figure()
    for i in range(A.shape[1]):
        
        if np.max(Qz[:,i]) - np.min(Qz[:,i]) >  0.1 and np.min(A[:,i]) < 4e4:
            
            curvelist += [i,]
            #plt.figure()
            #plt.errorbar(Qz[:,i],A[:,i],sigma[:,i])
    #return Qz      
    x0, Amp, G = np.full((3,len(curvelist)),0.)
    ct = 0
    
    gss = np.array([0,np.max(A),1])
    if PlotResults:
        plt.figure()
    #return gss
    for i in curvelist:
        
        try:
            ft = curve_fit(Lor,Qz[:,i],A[:,i], p0 = gss)
            print('Good Fit!')
        except:
            ft = (gss,gss)
            print('Bad Fit!')
        gss = ft[0]
        x0[ct] = gss[0]
        #A0[ct] = gss[1]
        Amp[ct] = gss[1]
        G[ct] = gss[2]
        
        
        if PlotResults:
            plt.plot(Qz[:,i],A[:,i])
            plt.plot(Qz[:,i],Lor(Qz[:,i],x0[ct],Amp[ct],G[ct]))
        
        ct += 1
    #return x0, Amp, G, args
    GQz = np.mean(G)
    #return args
    gss = 0. + args[7]#0. + np.mean(args,axis=0)
    gss[6:18:2] = 0. + np.sum(A * args[:,6:18:2],axis=0) / np.sum(A,axis = 0)
    gss[7:19:2] = 0. + np.sum(A * args[:,7:19:2],axis=0) / np.sum(A,axis = 0)
    gss[19:25] = 0. + np.sum(A * args[:,19:25],axis=0) / np.sum(A,axis = 0)
    #gss[6:] = 0. + np.sum(A * args[:,6:], axis=0) / np.sum(A,axis=0)
    Na = args.shape[1]
    
    Npks = int((Na-2)/4)
    
    gss[:Npks] = 1
    
    gss[-1] = GQz
    
    #gss[-8:-2] = np.mean(gss[-8:-2])
    
    Qxy = np.full(dims,0.)
    Qxy = (np.roll(np.linspace(0,dims[0]-1,dims[0])-int(dims[0]/2),int(dims[0]/2))[:,None,None]/2 \
           + Qxy,np.roll(np.linspace(0,dims[1]-1,dims[1])-int(dims[1])/2,int(dims[1]/2))[None,:,None] \
               + Qxy,np.roll(np.linspace(0,dims[2]-1,dims[2])-int(dims[2]/2),int(dims[2]/2))[None,None,:] + Qxy)
    
    tsn = IQxyzFt(Qxy,gss)
    
    return tsn, gss
    
    
def Lor(x,x0,A,G):
    
    return A / ((x-x0)**2 / G**2 +1)


def FitRawv2(fIm, sigma = None, cent = 7, PeakNum = 6):
    
    Nxy = fIm.shape[1] 
    Qx = np.linspace(0,Nxy-1,Nxy)
    Qxy = np.full((Nxy,Nxy),0.)
    Qxy = (Qx[:,None] + Qxy,Qx[None,:] + Qxy)
    Qxyf = (Qxy[0].flatten(),Qxy[1].flatten())
    Ag = np.full(PeakNum,1) #np.array([1,1,1,1,1,1])
    
    fImcf = ndimage.gaussian_filter(fIm[cent],5)
    
    pks = np.full((PeakNum,2),0.)
    
    for i in range(PeakNum):
        
        pks[i] = np.array(np.where(fImcf == np.max(fImcf))).reshape(2)
        fImcf[(Qxy[0] - pks[i,0])**2 + (Qxy[1] - pks[i,1])**2 < 10**2] = 0
    
    th = np.arctan2(pks[:,0] - np.mean(pks[:,0]), pks[:,1]-np.mean(pks[:,1]))
    pks = pks[th.argsort()[::-1]]
    #return pks, th

    gss = (np.quantile(fIm[cent],0.9999)*Ag).flatten().tolist() + pks.flatten().tolist() + (0.75*Ag).flatten().tolist() + [0.75,0.75]
    #gss0 = gss.copy()
    
    ftob = curve_fit(lambda Qxyf, *gss: IQxyFt(Qxyf,gss), Qxyf, fIm[cent].flatten(), p0=gss, sigma = sigma[cent].flatten())
    #return ftob, IQxyFt(Qxy,ftob[0]), gss, IQxyFt(Qxy,gss)
    #return gss, IQxyFt(Qxy,gss)
    gss0 = 0. + ftob[0]
    gss = 0. + ftob[0]
    ftprms = np.full((fIm.shape[0],PeakNum*4 + 2),0.)
    ftunc = 0. * ftprms
    
    Ft = 0. * fIm
    
    #return gss, IQxyFt(Qxy,gss)
    fails = []
    for j in tqdm(range(fIm.shape[0])):
        '''
        for i in range(6):
            
            gss[i] = np.max(fIm[j,int(gss[6+2*i])-2:2+int(gss[6+2*i]),int(gss[7+2*i])-2:2+int(gss[7+2*i])])
            gss[i-8] *= 1.5
        '''
        #gss[-2] *= 1.5
        #gss[:6] = np.max(fIm[j,])
        #Ft[j] = IQxyFt(Qxy,gss)
        
        #return gss, IQxyFt(Qxy,gss)
        try:
            ftob = curve_fit(lambda Qxyf, *gss: IQxyFt(Qxyf,gss), Qxyf, fIm[j].flatten(), p0=gss, sigma = sigma[j].flatten())
        except:
            try:
                
                gss = 0. + gss0
                
                for i in range(PeakNum):
                    
                    gss[i] = np.max(fIm[j,int(gss[PeakNum+2*i])-PeakNum + 1:PeakNum - 1 +int(gss[PeakNum+2*i]),
                                        int(gss[PeakNum + 1+2*i])-PeakNum + 1:PeakNum - 1 + int(gss[PeakNum + 1+2*i])])
                
                ftob = curve_fit(lambda Qxyf, *gss: IQxyFt(Qxyf,gss), Qxyf, fIm[j].flatten(), p0=gss, sigma = sigma[j].flatten())
                
                
                
            except:
                
                fails += [j,]
        
        ftprms[j] = ftob[0]
        gss = 0. + ftob[0]
        ftunc[j] = ftob[1].diagonal()
        #gss = ftob[0]
        Ft[j] = IQxyFt(Qxy,ftob[0])
    
    if len(fails) == 0:
        print('All fits converged...')
    else:
        print('Fits for projections ' + str(fails) + ' failed!')
    return ftprms, ftunc, Ft

def FitRaw(fIm, Qr0, Q0xy, sigma = None, cent = 7):
    Nxy = fIm.shape[1] 
    Qx = np.linspace(0,Nxy-1,Nxy)
    Qxy = np.full((Nxy,Nxy),0.)
    Qxy = (Qx[:,None] + Qxy,Qx[None,:] + Qxy)
    Qxyf = (Qxy[0].flatten(),Qxy[1].flatten())
    
    Ag = np.array([1,1,1,1,1,1])
    
    
    
    #Qr0 = 15.2;
    Q0g = np.array([[Qr0,0],[-Qr0,0],[Qr0*0.5,Qr0*np.sqrt(3)/2],[-Qr0*0.5,Qr0*np.sqrt(3)/2],[Qr0*0.5,-Qr0*np.sqrt(3)/2],[-Qr0*0.5,-Qr0*np.sqrt(3)/2]])
    Q0g[:,0] += Q0xy[0]
    Q0g[:,1] += Q0xy[1]
    
    gss = (np.quantile(fIm[cent],0.9999)*Ag).flatten().tolist() + Q0g.flatten().tolist() + (1.5*Ag).flatten().tolist() + [1.5,1.5]
    #gss0 = gss.copy()
    
    ftob = curve_fit(lambda Qxyf, *gss: IQxyFt(Qxyf,gss), Qxyf, fIm[cent].flatten(), p0=gss, sigma = sigma[cent].flatten())
    return ftob, IQxyFt(Qxy,ftob[0]), gss, IQxyFt(Qxy,gss)
    gss0 = 0. + ftob[0]
    gss = 0. + ftob[0]
    ftprms = np.full((fIm.shape[0],26),0.)
    ftunc = 0. * ftprms
    
    Ft = 0. * fIm
    
    #return gss, IQxyFt(Qxy,gss)
    fails = []
    for j in tqdm(range(fIm.shape[0])):
        '''
        for i in range(6):
            
            gss[i] = np.max(fIm[j,int(gss[6+2*i])-2:2+int(gss[6+2*i]),int(gss[7+2*i])-2:2+int(gss[7+2*i])])
            gss[i-8] *= 1.5
        '''
        #gss[-2] *= 1.5
        #gss[:6] = np.max(fIm[j,])
        #Ft[j] = IQxyFt(Qxy,gss)
        
        #return gss, IQxyFt(Qxy,gss)
        try:
            ftob = curve_fit(lambda Qxyf, *gss: IQxyFt(Qxyf,gss), Qxyf, fIm[j].flatten(), p0=gss, sigma = sigma[j].flatten())
        except:
            try:
                
                gss = 0. + gss0
                
                for i in range(6):
                    
                    gss[i] = np.max(fIm[j,int(gss[6+2*i])-5:5+int(gss[6+2*i]),int(gss[7+2*i])-5:5+int(gss[7+2*i])])
                
                ftob = curve_fit(lambda Qxyf, *gss: IQxyFt(Qxyf,gss), Qxyf, fIm[j].flatten(), p0=gss, sigma = sigma[j].flatten())
                
                
                
            except:
                
                fails += [j,]
        
        ftprms[j] = ftob[0]
        gss = 0. + ftob[0]
        ftunc[j] = ftob[1].diagonal()
        #gss = ftob[0]
        Ft[j] = IQxyFt(Qxy,ftob[0])
    if len(fails) == 0:
        print('All fits converged...')
    else:
        print('Fits for projections ' + str(fails) + ' failed!')
    return ftprms, ftunc, Ft


def FitLor(x,A,x0,G):
    
    return A / ((x-x0)**2 / G**2 + 1)

def SpK(Qz,Qx,Qy,Ixy,Gr,Gth):
    
    return Ixy[None,:] / ((Qz[:,None,None]**2 + Qx[None,:,None]**2 + Qy[None,None,:]**2)/Gr**2+1)

'''
def IqA(QzS,A,Q0,Gxy,Gr,Gz):
    
    Qz = QzS[2]
    Qy = QzS[1]
    Qx = QzS[0]
    
    rtn = np.full(Qz.shape,0.)
    
    Qc = np.full(2,0.)
    Qc = np.mean(Q0,axis = 0)
    
    
    for j in range(Q0.shape[0]): 
        Qr = ((Q0[j,0]-Qc[0])**2 + (Q0[j,1]-Qc[1])**2)**0.5
        rtn += A[j] / ((Qx-Q0[j,0]+Qc[0])**2 / Gxy[j]**2 + (Qy-Q0[j,1]+Qc[1])**2 / Gxy[j]**2 + Qz**2 / Gz**2 + 1) / ((((Qx)**2 + (Qy)**2 + Qz**2)**0.5-Qr)**2 / Gr**2 + 1)
        
    return rtn
'''
def IqA(QzS,A,Q0,Gxy,Gr,Gz):
    
    Qz = QzS[2]
    Qy = QzS[1]
    Qx = QzS[0]
    
    rtn = np.full(Qz.shape,0.)
    
    Qc = np.full(2,0.)
    Qc = np.mean(Q0,axis = 0)
    
    
    for j in range(Q0.shape[0]): 
        Qr = ((Q0[j,0]-Qc[0])**2 + (Q0[j,1]-Qc[1])**2)**0.5
        rtn += A[j] / ((Qx-Q0[j,0]+Qc[0])**2 / Gxy[j]**2 + (Qy-Q0[j,1]+Qc[1])**2 / Gxy[j]**2 + Qz**2 / Gz**2 + 1) / ((((Qx)**2 + (Qy)**2 + Qz**2)**0.5-Qr)**2 / Gr**2 + 1)
        
    return rtn

def IqAFt(QzS,*args):
    
    
    Na = len(args[0][:])
    
    Npks = int((Na-2)/4)
    
    A = np.array(args[0][:Npks])
    Q0 = np.array(args[0][Npks:3*Npks]).reshape((Npks,2))
    Gxy = args[0][3*Npks:4*Npks]
    Gr = args[0][4*Npks]
    Gz = args[0][4*Npks + 1]
    
    return IqA(QzS,A,Q0,Gxy,Gr,Gz)

def QzSlice(dims, theta, psi, spct = 1, wavelength = 6e-8, dx = 14.7e-7, dth = 0.4/180*np.pi):
    
    Nxyz = dims[0]*dims[1]*dims[2]
    Nxy = theta.shape[0]*dims[1]*dims[2]
    Nz = dims[0]
    Nx = dims[1]
    Ny = dims[2]
    
    qx = np.linspace(0,dims[1]-1,dims[1]) - dims[1]/2
    qy = np.linspace(0,dims[2]-1,dims[2]) - dims[2]/2
    
    qx = np.roll(qx, -int(dims[1]/2))
    qy = np.roll(qy, -int(dims[2]/2))
    
    k0 = 2*np.pi/wavelength * dx
    
    k0x = 2*np.pi*np.sin(theta)/wavelength*dx#/dims[1]*kx
    k0y = 2*np.pi*np.sin(psi)*np.cos(theta)/wavelength*dx#/dims[2]*ky
    
    k0z = np.sqrt(k0**2 - k0x**2 - k0y**2)
    
    qxp = (np.cos(theta[:,None,None]) * qx[None,:,None] - np.sin(theta[:,None,None]) * np.sin(psi[:,None,None]) * qy[None,None,:])
    qyp = ( np.cos(psi[:,None,None]) * qy[None,None,:] + 0. * qx[None,:,None] )
    #qz = (np.sin(theta[:,None,None])*qx[None:,None] + np.cos(theta[:,None,None]) * np.sin(psi[:,None,None]) * qy[None,None,:])/spct
    #qz = (((k0**2 - (k0x[:,None,None] + qx[None,:,None]*np.pi*2/dims[1])**2 - (k0y[:,None,None] + qy[None,None,:]*np.pi*2/dims[2])**2)**0.5 - k0z[:,None,None])/spct/(np.pi*2)*np.sqrt(dims[1]*dims[2])).flatten() #(np.sin(theta)*qx[:,None] + np.cos(theta) * np.sin(psi) * qy[None,:]).flatten()/spct
    qz = (((k0**2 - (k0x[:,None,None] + qx[None,:,None]*np.pi*2/dims[1])**2 - (k0y[:,None,None] + qy[None,None,:]*np.pi*2/dims[2])**2)**0.5 - k0z[:,None,None])*spct/(np.pi*2)*dims[0]) #(np.sin(theta)*qx[:,None] + np.cos(theta) * np.sin(psi) * qy[None,:]).flatten()/spct
    #print(np.max(qz))
    return qz, qxp, qyp

def QzSliceC(dims, theta, psi, Q00, spct = 1, wavelength = 6e-8, dx = 14.7e-7, dth = 0.4/180*np.pi):
    
    Nxyz = dims[0]*dims[1]*dims[2]
    Nxy = theta.shape[0]*dims[1]*dims[2]
    Nz = dims[0]
    Nx = dims[1]
    Ny = dims[2]
    
    qx = np.linspace(0,dims[1]-1,dims[1]) - Q00[0] #dims[1]/2
    qy = np.linspace(0,dims[2]-1,dims[2]) - Q00[1] #dims[2]/2
    
    #qx = np.roll(qx, -int(dims[1]/2))
    #qy = np.roll(qy, -int(dims[2]/2))
    
    k0 = 2*np.pi/wavelength * dx
    
    k0x = 2*np.pi*np.sin(theta)/wavelength*dx#/dims[1]*kx
    k0y = 2*np.pi*np.sin(psi)*np.cos(theta)/wavelength*dx#/dims[2]*ky
    
    k0z = np.sqrt(k0**2 - k0x**2 - k0y**2)
    
    qxp = (np.cos(theta[:,None,None]) * qx[None,:,None] - np.sin(theta[:,None,None]) * np.sin(psi[:,None,None]) * qy[None,None,:])
    qyp = ( np.cos(psi[:,None,None]) * qy[None,None,:] + 0. * qx[None,:,None] )
    #qz = (np.sin(theta[:,None,None])*qx[None:,None] + np.cos(theta[:,None,None]) * np.sin(psi[:,None,None]) * qy[None,None,:])/spct
    #qz = (((k0**2 - (k0x[:,None,None] + qx[None,:,None]*np.pi*2/dims[1])**2 - (k0y[:,None,None] + qy[None,None,:]*np.pi*2/dims[2])**2)**0.5 - k0z[:,None,None])/spct/(np.pi*2)*np.sqrt(dims[1]*dims[2])).flatten() #(np.sin(theta)*qx[:,None] + np.cos(theta) * np.sin(psi) * qy[None,:]).flatten()/spct
    qz = (((k0**2 - (k0x[:,None,None] + qx[None,:,None]*np.pi*2/dims[1])**2 - (k0y[:,None,None] + qy[None,None,:]*np.pi*2/dims[2])**2)**0.5 - k0z[:,None,None])*spct/(np.pi*2)*dims[0]) #(np.sin(theta)*qx[:,None] + np.cos(theta) * np.sin(psi) * qy[None,:]).flatten()/spct
    #print(np.max(qz))
    return qz, qxp, qyp
#gss = (7e3*Ag).flatten().tolist() + Q0g.flatten().tolist() + (3*Ag).flatten().tolist() + [1,]
#ftob = curve_fit(lambda tsf, *gss: IqAFt(tsf,gss), tsf, sg.flatten(), p0=gss)

'''
x0 = 6.37528665e+01
y0 = 6.54205440e+01

fx = int(np.floor(x0))
fy = int(np.floor(y0))
fI0 = 0. * EmptBm
fI0 += (1 + fx - x0)*(1 + fy - y0) * np.roll(EmptBm,(-fx,-fy),axis = (0,1))
fI0 += (- fx + x0)*(1 + fy - y0) * np.roll(EmptBm,(-1-fx,-fy),axis = (0,1))
fI0 += (1 + fx - x0)*(- fy + y0)* np.roll(EmptBm,(-fx,-1-fy),axis = (0,1))
fI0 += (- fx + x0)*(- fy + y0)* np.roll(EmptBm,(-1-fx,-1-fy),axis = (0,1))
#fI0 *= ArRat * np.exp(-SigT) 
#fI0 *= 0.026920275270886914

ttl = np.sum(fI0)
fI0 *= 0.
fI0[0,0] = ttl
fI0 = ndimage.gaussian_filter(fI0,1.,mode = 'wrap') * ArRat * np.exp(-SigT)


fIm = np.clip(SANS-Flare[None,:],0,None) #+ ArRat * np.exp(-SigT) * EmptBm[None,:]


#x0 = 64.21961233
#y0 = 65.04392726

x0 = 64.14901146
y0 = 65.04046213

fx = int(np.floor(x0))
fy = int(np.floor(y0))

#fIm = fIm + fI0 
fIm = ((1 + fx - x0)*(1 + fy - y0) * np.roll(fIm,(-fx,-fy),axis = (1,2))
       + (- fx + x0)*(1 + fy - y0) * np.roll(fIm,(-1-fx,-fy),axis = (1,2))
       + (1 + fx - x0)*(- fy + y0)* np.roll(fIm,(-fx,-1-fy),axis = (1,2))
       + (- fx + x0)*(- fy + y0)* np.roll(fIm,(-1-fx,-1-fy),axis = (1,2)))

rr = np.roll(np.linspace(0,127,128) - 64,64)
rs = rr[:,None]**2 + rr[None,:]**2
rs0 = 11
#fIm *= np.sign(np.clip(rs-11**2,0,1))

fIm += (1 - np.sum(fIm,axis=(1,2))[:,None,None]/np.sum(fI0)) * fI0
sigm = SANSUnc**2 + FlareUnc[None,:]**2 + 1e6*(ArRat * np.exp(-SigT) * np.clip(EmptBmUnc[None,:] - 4e3,0,None))**2
sigm = ((1 + fx - x0)*(1 + fy - y0) * np.roll(sigm,(-fx,-fy),axis = (1,2))
       + (- fx + x0)*(1 + fy - y0) * np.roll(sigm,(-1-fx,-fy),axis = (1,2))
       + (1 + fx - x0)*(- fy + y0)* np.roll(sigm,(-fx,-1-fy),axis = (1,2))
       + (- fx + x0)*(- fy + y0)* np.roll(sigm,(-1-fx,-1-fy),axis = (1,2)))
wts = 1 / np.maximum(9,sigm) * np.sign(np.clip(rs-rs0**2,0,1))

wts = wts.astype(np.float32)
wts0 = 0.* fIm +  np.sign(np.clip(rs-rs0**2,0,1)) # / np.sum( np.sign(np.clip(rs-rs0**2,0,1)) * fIm)

wts0 = wts0.astype(np.float32)

ProjPsi = Proj + 0.
ProjTh = 0. * Proj

fIm = fIm.astype(np.float32)
fI0 = fI0.astype(np.float32)


hf = h5py.File('SANSProcessed.h5')
hf.create_dataset('fIm',shape = fIm.shape, dtype = np.float32, data = fIm)
hf.create_dataset('fI0',shape = fI0.shape, dtype = np.float32, data = fI0)
hf.create_dataset('wts',shape = wts.shape, dtype = np.float32, data = wts)
hf.create_dataset('wts0',shape = wts0.shape, dtype = np.float32, data = wts0)
hf.create_dataset('ProjPsi',shape = ProjPsi.shape, dtype = np.float32, data = ProjPsi)
hf.create_dataset('ProjTh',shape = ProjTh.shape, dtype = np.float32, data = ProjTh)
hf.close()
'''