# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:07:11 2022

@author: bjh3
"""
import h5py
import numpy as np
import scipy.ndimage as ndimage

def FileExtract(flnm, zoom, pad, QTrue = False, rotate = False, order = 0):
    
    hf = h5py.File(flnm, 'r')
    I = hf['_slice_1/main/I(QxQy)/I'][:]
    sigI = hf['_slice_1/main/I(QxQy)/Idev'][:]
    
    
    I = np.pad(ndimage.zoom(I, zoom, order = order), pad)
    sigI = np.pad(ndimage.zoom(sigI**2, zoom, order = order)**0.5, pad) #This does NOT account for correlations!
    
    if rotate:
        
        I = ndimage.rotate(I,90)
        sigI = ndimage.rotate(sigI,90)
    
    if QTrue:
        Qx = hf['_slice_1/main/I(QxQy)/Qx'][:]
        Qy = hf['_slice_1/main/I(QxQy)/Qy'][:]
        Qx = np.pad(ndimage.zoom(Qx, zoom, order = order), pad)
        Qy = np.pad(ndimage.zoom(Qy, zoom, order = order), pad)
        
        wvlgsprd = float(hf['reduction_information/sample_logs/main/wavelength_spread'][()][1:6])
        wvlg = float(hf['reduction_information/sample_logs/main/wavelength'][()][1:6]) * 1e-8
        
        
        hf.close()
        
        if rotate:
            
            Qx = ndimage.rotate(Qx,90)
            Qy = ndimage.rotate(Qy,90)
    
        return I, sigI, Qx, Qy, wvlg, wvlgsprd
    
    else:
        
        hf.close()
        return I, sigI
    
def ComputeZoomPad(filename, dimscatt = (128,128)):
    
    hf = h5py.File(filename, 'r')
    I = hf['_slice_1/main/I(QxQy)/I'][:]
    dimscatt_raw = I.shape[-2:]
    Qx = hf['_slice_1/main/I(QxQy)/Qx'][:]
    Qy = hf['_slice_1/main/I(QxQy)/Qy'][:]
    mid = (int(dimscatt_raw[0]/2),int(dimscatt_raw[1]/2))
    dQx = np.abs(Qx[mid[0]+1,mid[1]+1] - Qx[mid[0],mid[1]])
    dQy = np.abs(Qy[mid[0]+1,mid[1]+1] - Qy[mid[0],mid[1]])
    
    if dimscatt_raw[0] * dQx > dimscatt_raw[1] * dQy:
        
        zoomx = dimscatt[0] / dimscatt_raw[0]
        Ny = np.round(dimscatt_raw[1] * dQy / dQx * zoomx)
        zoomy = Ny / dimscatt_raw[1]
        padx = (0,0)
        padyL = int((dimscatt[1]-Ny)/2)
        padyR = int(dimscatt[1] - Ny - padyL)
        pady = (padyL,padyR)
        
    else:
        
        zoomy = dimscatt[1] / dimscatt_raw[1]
        Nx = np.round(dimscatt_raw[0] * dQx / dQy * zoomy)
        zoomx = Nx / dimscatt_raw[0]
        pady = (0,0)
        padxL = int((dimscatt[0]-Nx)/2)
        padxR = int(dimscatt[0] - Nx - padxL)
        padx = (padxL,padxR)
    
    dQ = (dQx / zoomx + dQy / zoomy)/2
    dx = 2 * np.pi /np.sqrt(dimscatt[0] * dimscatt[1]) / dQ * 1e-8
        
    return (zoomx, zoomy), (padx,pady), dx
        
    

ProjThHFIR = np.array([0,]*21 + np.arange(-5,0).tolist() + np.arange(1,6).tolist())
ProjPsiHFIR = np.array(np.arange(-10,11).tolist() + [0,] * 10)

filenumbers = np.arange(47682,47713).tolist()

prefix = 'D:/OakRidge/2mmAp/no_sample_transmission'
suffix = '_reduction_log.hdf'

files = []

emptfile = prefix + str(47784) + suffix

for i in filenumbers:
    
    files += [prefix + str(i) + suffix]

def HFIRtoTomo(files, emptfile, ProjTh = ProjThHFIR, ProjPsi = ProjPsiHFIR, 
               savename = 'SANS.h5', dimscatt = (128,128), attn = 0.2684, 
               rotate = True, order = 0, M0 = 1900, H0 = 250, Npks = [6,2,2], 
               SampleDims = [0.33,0.31,0.30], wavelengthdist = 'triangular'):
    
    zoom, pad, dx = ComputeZoomPad(files[0], dimscatt)
    
    ProjNum = 0
    
    for i in files:
                
        #flnm = prefix + str(i) + suffix

        if ProjNum == 0:
            
            I, sigI, Qx, Qy, wvlg, wvlgsprd = FileExtract(i, zoom, pad, True, rotate, order = order)
            
            SANSshape = (2,) + np.array(files).shape + I.shape
            
            SANS, SANSUnc = np.full(SANSshape, 0., dtype = np.float32)
        
        else:
            
            I, sigI = FileExtract(i, zoom, pad, False, rotate, order = order)
            
        SANS[ProjNum] = np.clip(I,0,None)
        SANSUnc[ProjNum] = sigI
        
        ProjNum += 1
        
    #flnm = prefix + str(emptfile) + suffix
    I, sigI = FileExtract(emptfile, zoom, pad, False, rotate, order = order)
    I = np.clip(I,0,None)
    I *= 1/attn
    sigI *= 1/attn
    #print(wvlg)
    sv = h5py.File(savename, 'w')
    
    sv.create_dataset('EmptBm',shape = I.shape, dtype = np.float32, data = I)
    sv.create_dataset('EmptBmUnc',shape = sigI.shape, dtype = np.float32, data = sigI)
    sv.create_dataset('ProjTh',shape = ProjTh.shape, dtype = np.float32, data = ProjTh)
    sv.create_dataset('ProjPsi',shape = ProjPsi.shape, dtype = np.float32, data = ProjPsi)
    sv.create_dataset('SANS',shape = SANS.shape, dtype = np.float32, data = SANS)
    sv.create_dataset('SANSUnc',shape = SANSUnc.shape, dtype = np.float32, data = SANSUnc)
    sv.create_dataset('wavelength', data = wvlg)
    sv.create_dataset('wavelength_spread', data = wvlgsprd)
    sv.create_dataset('dx',data = dx)
    sv.create_dataset('M0',data = M0)
    sv.create_dataset('H0',data = H0)
    sv.create_dataset('Npks',data = Npks)
    sv.create_dataset('SampleDims',data = SampleDims)
    sv.create_dataset('wavelengthdist',data = wavelengthdist)
    
    sv.close()
    
    