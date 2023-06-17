# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:45:46 2021

@author: bjh3
"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

from tqdm import tqdm

from scipy import ndimage

from scipy.ndimage import gaussian_filter

import pyfftw

import time

import h5py

####
# Note changed order of operations for Fwrd, FwrdRdm, dChidM
# old lines ### before
# new lines #new# after


class PropParams:
    
    def __init__(self, shape, ProjTh = np.array(None), ProjPsi = np.array(None), fI0 = np.array(None), N = np.array([1,1,1]), wavelength = 6e-8, 
                 dx = 14.7E-7, xFd = np.array(None), Fd = np.array(None), dimscatt = None, Paxis = np.array(None), PsiSwitch = 45, ThetaSwitch = 45, spct = [1,1,1]):
        
        self.fftm, self.ifftm, self.a, self.fa, self.fftup, self.ifftup, \
            self.aup, self.faup = MakeFFTW3D(shape)
        
        self.fI0 = fI0
        self.ProjTh = ProjTh
        self.ProjPsi = ProjPsi            
        self.wavelength = wavelength
        self.dx = dx
        self.N = N
        
        if Paxis.all() == None:
            Paxis = (0 * ProjTh).astype(int)
            Paxis += np.sign(np.clip(np.abs(ProjTh) - ThetaSwitch,0,None)).astype(int)
            Paxis += 2 * np.sign(np.clip(np.abs(ProjPsi) - PsiSwitch,0,None)).astype(int)
        
        self.Paxis = Paxis
        
        if xFd.all() == None:
            xFd = np.linspace(-0.1,0.1,41)
            Fd = 0.1 - np.abs(xFd)
            Fd = Fd / np.sum(Fd)
        
        if dimscatt == None:
            if fI0.all() == None:
                dimscatt = shape[2:4]
            else:
                dimscatt = (fI0.shape[-2],fI0.shape[-1])
            
        self.dimscatt = dimscatt
        
        self.Y, self.YT = MakeY(dimscatt, Fd, xFd)
        
        self.W, self.WT = WprojA(shape, ProjTh, ProjPsi, dimscatt, Paxis, spct = spct)
        
        #self.Y, self.YT = MakeY3D(shape, Fd, xFd)
        
        #self.W, self.WT = MakeW3D(shape, ProjTh, ProjPsi, dimscatt)
        

def FwrdRdm(ProjTh, ProjPsi, fI0, mdims = (3,512,128,128), spct = 1, 
            wavelength = 6e-8, dx = 14.7E-7, M0 = 1900, H0 = 250, N = 1, 
            flpth = 'UberSim/simulation_ubermag_large_', flsuff = '.hdf5', Nsm = 10, Nz = 26, 
            flst = 1, xFd = np.array(None), Fd = np.array(None)):
    '''
    Propagates neutron spinor through 

    Parameters
    ----------
    ProjTh : 1D numpy array of size (Nproj,).  
        Projections along the horizontal axis, [deg].
    ProjPsi : 1D numpy array of size (Nproj,). 
        Projections about the vertical axis, [deg].
    fI0 : 2D numpy array. 
        Normalized incoming neutron beam intensity.
    mdims : Tuple of length 4. optional
        Size of magnetization array. (3,Nz,Nx,Ny). The default is (3,512,128,128).
    spct : Float or int, optional
        Voxel height / voxel width. The default is 1.
    wavelength : Float, optional
        Wavelength in cm. The default is 6e-8.
    M0 : Float, optional
        4 pi Ms, [G]. The default is 1900.
    H0 : Float, optional
        Guide field, [G]. The default is 250.
    N : Float or int, optional
        Number of self-convolutions to perform per file. Total sample 
        thickness is N * Nsm. The default is 1.
    flpth : String, optional
        File path and prefix for magnetization. Assumes that magnetization is
        saved under 'field/array' in (Nx,Ny,Nz,n) format. The default is 
        'UberSim/simulation_ubermag_large_'.
    Nsm : Int, optional
        Number of files from which to pull magnetizations. File names should 
        end with number between 'flst' and 'flst + Nsm'. The default is 10.
    Nz : Int, optional
        The number of magnetization files through which the neutron propagates. 
        The default is 26.
    flst : Int, optional
        File number starting index. The default is 1.
    Fd : Numpy array, 1D, optional
        dk / k0 wavelength distribution. The default results in a triangular 
        distribution determined by xFd.
    xFd : Numpy array 1D, optional
        dk / k0 x-values for Fd, [rad]. The default is an equally-spaced array
        of length 41 with range +/- 0.1 rad.

    Returns
    -------
    s : Unpolarized SANS pattern, including effects of incoming beam and 
        wavelength  distribution.
    ss : Set of (Nproj, 2, 2, Nqx, Nqy) spin-dependent scattering matrices. 
        Does NOT include N-convolutions, wavelength dist (Fd, xFd), or beam 
        tilting (see Wproj() function).

    '''
    
    m = np.full(mdims,0.)
    #fftm, ifftm, a, fa, fftup, ifftup, aup, faup = MakeFFTW(m.shape)
    X = PropParams(mdims, ProjTh, ProjPsi, fI0, N, wavelength, dx, xFd, Fd)
    
    
    dims = m.shape[2:4]
    
    q = MakeQ(m.shape,spct = spct)
    
    I0 = np.fft.fftn(fI0,axes=(-2,-1))
    
    s = np.full((ProjTh.shape[0],dims[0],dims[1]),0.)
    ss = np.full((ProjTh.shape[0],2,2,dims[0],dims[1]),0. + 0.j)
    
    PsiUp = np.full((m.shape[1],m.shape[2],m.shape[3],2),0. + 0.j)
    PsiDn = np.full((m.shape[1],m.shape[2],m.shape[3],2),0. + 0.j)
    
    
    for i in range(ProjTh.shape[0]):
        
        print('Computing scattering pattern for projection ' + str(i+1) + ' of ' + str(ProjTh.shape[0]) + '...')
        
        PsiUp[-1,:,:,0] = 1.
        PsiDn[-1,:,:,1] = 1.
        PsiUp[-1,:,:,1] = 0.
        PsiDn[-1,:,:,0] = 0.
        
        for j in tqdm(range(Nz)):
            
            hf = h5py.File(flpth + str(flst + np.random.randint(0,Nsm)) + flsuff, 'r')
            
            rll = (np.random.randint(0,mdims[1]),
                   np.random.randint(0,mdims[2]),
                   np.random.randint(0,mdims[3]))
            
            m[0] = np.roll(hf['field/array'][:].transpose(3,2,0,1)[2],rll,axis=(0,1,2))
            m[1] = np.roll(hf['field/array'][:].transpose(3,2,0,1)[0],rll,axis=(0,1,2))
            m[2] = np.roll(hf['field/array'][:].transpose(3,2,0,1)[1],rll,axis=(0,1,2))
            
            hf.close()
            
            B = BfromM(m, q, X)
            B *= M0
            B[0] += H0
            
            PsiAdv(PsiUp, PsiDn, B, ProjTh[i], ProjPsi[i], X, spct = spct)
            
            #PsiAdv(PsiUp, PsiDn, B,ProjTh[i],ProjPsi[i],fftup, ifftup, aup, 
            #       faup, spct = spct, wavelength = wavelength)
        ss[i] = np.stack([[np.fft.ifftn(PsiUp[-1,:,:,0]),
                           np.fft.ifftn(PsiDn[-1,:,:,0])],
                          [np.fft.ifftn(PsiUp[-1,:,:,1]),
                           np.fft.ifftn(PsiDn[-1,:,:,1])]])
        S = 0.5*np.sum(np.abs(ss[i])**2,axis=(0,1))
        
        S = np.fft.fftn(S) #new#
        
        s[i] = X.Y.dot(X.W[i].dot(np.real(np.fft.ifftn(S**X.N)).flatten())).reshape((m.shape[2],m.shape[3])) #new#
        
        if len(I0.shape) == 2:
        
            s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i]) * I0))
        
        else:
            s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i]) * I0[i]))
        
        
    return s, ss



def Fwrd(X, M, ld = True):
    #m, M0, H0, N, ProjTh, ProjPsi, fI0, spct = 1, wavelength = 6e-8, 
    #    sprd = 5e-3, siglam = 0.1, ld = True):
    
    #fftm, ifftm, a, fa, fftup, ifftup, aup, faup = MakeFFTW(M.m.shape)
    # now part of X
    
    
    M.ComputeB(X)
    
    I0 = np.fft.fftn(X.fI0,axes=(-2,-1))
    
    
    s = np.full((X.ProjTh.shape[0],) + X.dimscatt,0.)
    #s = []
    ss = []
    #ss = 0. * s
    
    PsiUp = np.full((M.m.shape[1],M.m.shape[2],M.m.shape[3],2),0. + 0.j)
    PsiDn = np.full((M.m.shape[1],M.m.shape[2],M.m.shape[3],2),0. + 0.j)
    
    
    if ld:
        rng = tqdm(range(X.ProjTh.shape[0]))
    else:
        rng = range(X.ProjTh.shape[0])
    
    for i in rng:
        
        if X.Paxis[i] == 0:
            PsiUp[-1,:,:,0] = 1.
            PsiDn[-1,:,:,1] = 1.
            PsiUp[-1,:,:,1] = 0.
            PsiDn[-1,:,:,0] = 0.
            #xyshape = (M.m.shape[2:4])
        elif X.Paxis[i] == 1:
            PsiUp[:,-1,:,0] = 1.
            PsiDn[:,-1,:,1] = 1.
            PsiUp[:,-1,:,1] = 0.
            PsiDn[:,-1,:,0] = 0.
            #xyshape = (M.m.shape[1],M.m.shape[3])
        else:
            PsiUp[:,:,-1,0] = 1.
            PsiDn[:,:,-1,1] = 1.
            PsiUp[:,:,-1,1] = 0.
            PsiDn[:,:,-1,0] = 0.
            #xyshape = (M.m.shape[1],M.m.shape[2])
        
        PsiAdv(PsiUp, PsiDn, M.B, X.ProjTh[i], X.ProjPsi[i], X, spct = M.voxelaspect, Paxis = X.Paxis[i])
    
        if X.Paxis[i] == 0:
            S = 0.5*(np.abs(np.fft.ifftn(PsiUp[-1,:,:,0]))**2 + np.abs(np.fft.ifftn(PsiUp[-1,:,:,1]))**2 
                 + np.abs(np.fft.ifftn(PsiDn[-1,:,:,0]))**2 + np.abs(np.fft.ifftn(PsiDn[-1,:,:,1]))**2)
        elif X.Paxis[i] == 1:
            S = 0.5*(np.abs(np.fft.ifftn(PsiUp[:,-1,:,0]))**2 + np.abs(np.fft.ifftn(PsiUp[:,-1,:,1]))**2 
                 + np.abs(np.fft.ifftn(PsiDn[:,-1,:,0]))**2 + np.abs(np.fft.ifftn(PsiDn[:,-1,:,1]))**2)
        else:
            S = 0.5*(np.abs(np.fft.ifftn(PsiUp[:,:,-1,0]))**2 + np.abs(np.fft.ifftn(PsiUp[:,:,-1,1]))**2 
                 + np.abs(np.fft.ifftn(PsiDn[:,:,-1,0]))**2 + np.abs(np.fft.ifftn(PsiDn[:,:,-1,1]))**2)
        #print('Angle: ' + str(ProjPsi[i]))
        #print(np.sum(S))
        #ss[i] = S + 0.
        ss += [S + 0.]
        #S *= 1 / np.sum(S)
        #old#S = np.fft.fftn(W[i].dot(S.flatten()).reshape((m.shape[2],m.shape[3])))
        S = np.fft.fftn(S) #new#
        #s00 = np.real(np.mean(S))
        #s00N = s00**N
        
        #old#s[i] = Y.dot(np.real(np.fft.ifftn(S**N)).flatten()).reshape((m.shape[2],m.shape[3]))

        s[i] = X.Y.dot(X.W[i].dot(np.real(np.fft.ifftn(S**X.N[X.Paxis[i]])).flatten())).reshape(X.dimscatt) #new#
        #s[i] = Y.dot(np.real(np.fft.ifftn(np.exp(N * np.log(S)))).flatten()).reshape((m.shape[2],m.shape[3]))
        #print(np.sum(s[i]))
        #s[i] = Y.dot(np.real(np.fft.ifftn(s00N * np.exp(N * (S/s00 - 1)))).flatten()).reshape((m.shape[2],m.shape[3]))
        if len(I0.shape) == 2:
            s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i]) * I0))
        else:
            s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i]) * I0[i]))
        #print(np.sum(s[i]))
        '''
        PsiUp[-1,:,:,0] = 1.
        PsiDn[-1,:,:,1] = 1.
        
        CS.PsiAdv(PsiUp, PsiDn, B,0., Proj[i], fftup, ifftup, aup, faup, spct = spct)
        
        Suu = PsiUp[-1,:,:,0]
        Sud = PsiUp[-1,:,:,1]
        Sdu = PsiDn[-1,:,:,0]
        Sdd = PsiDn[-1,:,:,1]
        
        S = 0.5*(np.abs(np.fft.ifftn(Suu))**2 + np.abs(np.fft.ifftn(Sdu))**2 
                 + np.abs(np.fft.ifftn(Sud))**2 + np.abs(np.fft.ifftn(Sdd))**2)
        
        S = np.fft.fftn(W[i].dot(S.flatten()).reshape(dims))
        #S = np.fft.fftn(S)

        s[i] = Y.dot(np.real(np.fft.ifftn(S**N)).flatten()).reshape(dims)
        
        s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i])*I0))
        '''
        
    return s, ss

# =============================================================================
# Derivative of ChiSq wrt m
# =============================================================================

def dChidM(X, M, ld = True):
    #(dchidm, m, M0, H0, fIm, ProjTh, ProjPsi, wts, I0, N, q, Y, YT, W, WT, 
    #       fftm, ifftm, a, fa, fftup, ifftup, aup, faup, 
    #       spct = 1, wavelength = 6e-8, pwr = 2, bkg = 1e-2, ld = True):
    
    '''
    Computes dChi / d m functional derivative
    
    # dchidm - spin density derivative (output)
    # m - spin density
    # M0 - saturated moment
    # H0 - guide field
    # fIm - measured diffraction intensity
    # I0 - fourier tranfrom of renormalized empty beam np.fft.fftn(fI0) 
    # I00 - fourier trasform of q=0 location np.fft.fftn(fI00)
    # N - number incoherent blocks in sample - Sample thickness / recon depth
    # q - unit vector array
    # Y, YT - wavelength spread transformation
    # W, WT - Free space wave projection operator
    # (i)fftm - 3D (inverse) fourier transform object
        - a - real space vector field
        - fa - Fourier space fector field
    # (i)fftup - 2D (inverse) fourier transform object
        - aup - real space spinor
        - faup - Fourier space spinor
    (optional)
    # spct - voxel aspect
    # wavelength - neutron wavelength, cm
    # ax - axis over which to rotate sample
    '''
    '''
    if ax == 0:
        
        ProjPsi = Projns + 0.
        Proj = 0.*Projns
    
    else:
        
        ProjPsi = 0*Projns
        Proj = 0. + Projns
    '''
    
    M.ComputeB(X)
    
    dims = M.B.shape
    #print(dims)
    #dchidm = np.full(m.shape,0.)
    
    X.dchidm *= 0.
    #dchidM0 *= 0.
    
    I0 = np.fft.fftn(X.fI0, axes = (-2,-1))
    
    s = np.full((X.ProjTh.shape[0],M.m.shape[2],M.m.shape[3]),0.)
    
    #Ikt = np.full((Proj.shape[0],m.shape[2],m.shape[3]),0. + 0.j)
    
    chisq = 0.
    
    
    PsiUp = np.full((dims[1],dims[2],dims[3],2),0. + 0.j, dtype = np.csingle)
    PsiDn = np.full((dims[1],dims[2],dims[3],2),0. + 0.j, dtype = np.csingle)
    
    ChiUp = np.full((dims[1],dims[2],dims[3],2),0. + 0.j, dtype = np.csingle)
    ChiDn = np.full((dims[1],dims[2],dims[3],2),0. + 0.j, dtype = np.csingle)
    
    
    if ld:
        rng = tqdm(range(X.ProjTh.shape[0]))
    else:
        rng = range(X.ProjTh.shape[0])
    
    for i in rng:
        
        #coeff = (1.91*5.05078E-24*dx*wavelength*mneutron)/(2*np.pi*(hbar**2))/np.cos(ProjTh[i]/180*np.pi)/np.cos(ProjPsi[i]/180*np.pi)
        
        coeff = 2.312e6 * X.dx * X.wavelength / np.cos(X.ProjTh[i]/180*np.pi) / np.cos(X.ProjPsi[i]/180*np.pi)
        
        #PsiUp[-1,:,:,0] = 1.
        #PsiDn[-1,:,:,1] = 1.
        #PsiUp[-1,:,:,1] = 0.
        #PsiDn[-1,:,:,0] = 0.
        
        
        if X.Paxis[i] == 0:
            PsiUp[-1,:,:,0] = 1.
            PsiDn[-1,:,:,1] = 1.
            PsiUp[-1,:,:,1] = 0.
            PsiDn[-1,:,:,0] = 0.
            xyshape = (M.m.shape[2:4])
        elif X.Paxis[i] == 1:
            PsiUp[:,-1,:,0] = 1.
            PsiDn[:,-1,:,1] = 1.
            PsiUp[:,-1,:,1] = 0.
            PsiDn[:,-1,:,0] = 0.
            xyshape = (M.m.shape[1],M.m.shape[3])
        else:
            PsiUp[:,:,-1,0] = 1.
            PsiDn[:,:,-1,1] = 1.
            PsiUp[:,:,-1,1] = 0.
            PsiDn[:,:,-1,0] = 0.
            xyshape = (M.m.shape[1],M.m.shape[2])
        
        PsiAdv(PsiUp, PsiDn, M.B, X.ProjTh[i], X.ProjPsi[i], X, spct = M.voxelaspect, Paxis = X.Paxis[i])
        #PsiAdv(PsiUp, PsiDn, M.B, X.ProjTh[i], X.ProjPsi[i], X, spct = M.voxelaspect)
        #(PsiUp, PsiDn, B,ProjTh[i],ProjPsi[i],fftup, ifftup, aup, faup, spct = spct, wavelength = wavelength)
        
        #S = 0.5*(np.abs(np.fft.ifftn(PsiUp[-1,:,:,0]))**2 + np.abs(np.fft.ifftn(PsiUp[-1,:,:,1]))**2 
        #         + np.abs(np.fft.ifftn(PsiDn[-1,:,:,0]))**2 + np.abs(np.fft.ifftn(PsiDn[-1,:,:,1]))**2)
        if X.Paxis[i] == 0:
            S = 0.5*(np.abs(np.fft.ifftn(PsiUp[-1,:,:,0]))**2 + np.abs(np.fft.ifftn(PsiUp[-1,:,:,1]))**2 
                 + np.abs(np.fft.ifftn(PsiDn[-1,:,:,0]))**2 + np.abs(np.fft.ifftn(PsiDn[-1,:,:,1]))**2)
        elif X.Paxis[i] == 1:
            S = 0.5*(np.abs(np.fft.ifftn(PsiUp[:,-1,:,0]))**2 + np.abs(np.fft.ifftn(PsiUp[:,-1,:,1]))**2 
                 + np.abs(np.fft.ifftn(PsiDn[:,-1,:,0]))**2 + np.abs(np.fft.ifftn(PsiDn[:,-1,:,1]))**2)
        else:
            S = 0.5*(np.abs(np.fft.ifftn(PsiUp[:,:,-1,0]))**2 + np.abs(np.fft.ifftn(PsiUp[:,:,-1,1]))**2 
                 + np.abs(np.fft.ifftn(PsiDn[:,:,-1,0]))**2 + np.abs(np.fft.ifftn(PsiDn[:,:,-1,1]))**2)
        #old#S = np.fft.fftn(W[i].dot(S.flatten()).reshape((m.shape[2],m.shape[3])))
        S = np.fft.fftn(S) #new#
        s[i] = X.Y.dot(X.W[i].dot(np.real(np.fft.ifftn(S**X.N[X.Paxis[i]])).flatten())).reshape(X.dimscatt)
        
        if len(I0.shape) == 2:
            s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i]) * I0))
        else:
            s[i] = np.real(np.fft.ifftn(np.fft.fftn(s[i]) * I0[i]))
        
        Rs = 2 * X.weights[i] * (s[i] - X.fIm[i])
        
        chisq += np.sum( X.weights[i] * (s[i] - X.fIm[i])**2)
        if len(I0.shape) == 2:
            Rs = np.real(np.fft.ifftn(np.conjugate(np.fft.ifftn(Rs) * I0))) #new#
        else:
            Rs = np.real(np.fft.ifftn(np.conjugate(np.fft.ifftn(Rs) * I0[i])))
        Rs = np.fft.fftn(X.WT[i].dot(X.YT.dot(Rs.flatten())).reshape(xyshape))*xyshape[0]*xyshape[1] #new#
        #Rs *= Rs.shape[0] * Rs.shape[1]
        Rs = X.N[X.Paxis[i]] * np.real(np.fft.ifftn(Rs * S**(X.N[X.Paxis[i]]-1))) #new#
        
        if X.Paxis[i] == 0:
            ChiUp[-1] = np.fft.ifftn(Rs[:,:,None] * np.conjugate(np.fft.ifftn(PsiUp[-1],axes=(0,1))),axes=(0,1))
            ChiDn[-1] = np.fft.ifftn(Rs[:,:,None] * np.conjugate(np.fft.ifftn(PsiDn[-1],axes=(0,1))),axes=(0,1))
        elif X.Paxis[i] == 1:
            ChiUp[:,-1] = np.fft.ifftn(Rs[:,:,None] * np.conjugate(np.fft.ifftn(PsiUp[:,-1],axes=(0,1))),axes=(0,1))
            ChiDn[:,-1] = np.fft.ifftn(Rs[:,:,None] * np.conjugate(np.fft.ifftn(PsiDn[:,-1],axes=(0,1))),axes=(0,1))
        else:
            ChiUp[:,:,-1] = np.fft.ifftn(Rs[:,:,None] * np.conjugate(np.fft.ifftn(PsiUp[:,:,-1],axes=(0,1))),axes=(0,1))
            ChiDn[:,:,-1] = np.fft.ifftn(Rs[:,:,None] * np.conjugate(np.fft.ifftn(PsiDn[:,:,-1],axes=(0,1))),axes=(0,1))
        
        #ChiRev(ChiUp, ChiDn, B,ProjTh[i],ProjPsi[i],fftup,ifftup,aup,faup,spct = spct, wavelength = wavelength)
        ChiRev(ChiUp, ChiDn, M.B, X.ProjTh[i], X.ProjPsi[i], X, M.voxelaspect, X.Paxis[i])
        
        
        X.a[0] = ChiUp[:,:,:,0] * PsiUp[:,:,:,0] - ChiUp[:,:,:,1] * PsiUp[:,:,:,1] + ChiDn[:,:,:,0] * PsiDn[:,:,:,0] - ChiDn[:,:,:,1] * PsiDn[:,:,:,1]
        X.a[1] = ChiUp[:,:,:,0] * PsiUp[:,:,:,1] + ChiUp[:,:,:,1] * PsiUp[:,:,:,0] + ChiDn[:,:,:,0] * PsiDn[:,:,:,1] + ChiDn[:,:,:,1] * PsiDn[:,:,:,0]
        X.a[2] = -1.j * ChiUp[:,:,:,0] * PsiUp[:,:,:,1] + 1.j*ChiUp[:,:,:,1] * PsiUp[:,:,:,0] - 1.j * ChiDn[:,:,:,0] * PsiDn[:,:,:,1] + 1.j * ChiDn[:,:,:,1] * PsiDn[:,:,:,0]
        
        X.a[:] = np.imag(X.a) # twice imaginary part, divided by two incoming spin states
        #a[:] *= 2
        X.fa = X.fftm()
    
        X.fa[:] += - M.qhat * np.sum(X.fa*M.qhat,axis = 0)
        
        X.a = X.ifftm()
        
        #if i ==0:
        #dchidM0[i] += coeff*M0*np.real(a) #/dims[3]
        X.dchidm += coeff*M.M0*np.real(X.a) #dchidM0[i]

    return chisq, s

def MakeFFTW3D(dims):
    
    '''
    Returns pyFFTW objects for 2D and 3D transforms, given reconstruction size
    
    # (i)fftm - 3D transform
        -- a - real space function
        -- fa - Fourier space function
        
    # (i)fftup - 2D spinor transform
        -- aup - real space spinor
        -- faup - Fourier space spinor
    '''
    
    a = pyfftw.empty_aligned(dims, dtype = 'complex64')
    fa = pyfftw.empty_aligned(dims, dtype = 'complex64')
    
    fftm = pyfftw.FFTW(a,fa,axes = (1,2,3))
    ifftm = pyfftw.FFTW(fa,a,axes = (1,2,3),direction='FFTW_BACKWARD')
    
    shp = (dims[2],dims[3],2)
    
    aup = [pyfftw.empty_aligned(shp, dtype = 'complex64')]
    faup = [pyfftw.empty_aligned(shp, dtype = 'complex64')]
    
    fftup = [pyfftw.FFTW(aup[0],faup[0],axes = (0,1),direction='FFTW_BACKWARD')]
    ifftup = [pyfftw.FFTW(faup[0],aup[0],axes = (0,1))]
    
    shp = (dims[1],dims[3],2)
    
    aup += [pyfftw.empty_aligned(shp, dtype = 'complex64')]
    faup += [pyfftw.empty_aligned(shp, dtype = 'complex64')]
    
    fftup += [pyfftw.FFTW(aup[1],faup[1],axes = (0,1),direction='FFTW_BACKWARD')]
    ifftup += [pyfftw.FFTW(faup[1],aup[1],axes = (0,1))]
    
    shp = (dims[1],dims[2],2)
    
    aup += [pyfftw.empty_aligned(shp, dtype = 'complex64')]
    faup += [pyfftw.empty_aligned(shp, dtype = 'complex64')]
    
    fftup += [pyfftw.FFTW(aup[2],faup[2],axes = (0,1),direction='FFTW_BACKWARD')]
    ifftup += [pyfftw.FFTW(faup[2],aup[2],axes = (0,1))]
    
    
    return fftm, ifftm, a, fa, fftup, ifftup, aup, faup

# =============================================================================
# Make FFTW Objects
# =============================================================================

def MakeFFTW(dims):
    
    '''
    Returns pyFFTW objects for 2D and 3D transforms, given reconstruction size
    
    # (i)fftm - 3D transform
        -- a - real space function
        -- fa - Fourier space function
        
    # (i)fftup - 2D spinor transform
        -- aup - real space spinor
        -- faup - Fourier space spinor
    '''
    
    a = pyfftw.empty_aligned(dims, dtype = 'complex64')
    fa = pyfftw.empty_aligned(dims, dtype = 'complex64')
    
    fftm = pyfftw.FFTW(a,fa,axes = (1,2,3))
    ifftm = pyfftw.FFTW(fa,a,axes = (1,2,3),direction='FFTW_BACKWARD')
    
    shp = (dims[2],dims[3],2)
    
    aup = pyfftw.empty_aligned(shp, dtype = 'complex64')
    faup = pyfftw.empty_aligned(shp, dtype = 'complex64')
    
    fftup = pyfftw.FFTW(aup,faup,axes = (0,1),direction='FFTW_BACKWARD')
    ifftup = pyfftw.FFTW(faup,aup,axes = (0,1))
    
    
    return fftm, ifftm, a, fa, fftup, ifftup, aup, faup
    

# =============================================================================
# 3D wave functions
# =============================================================================

def PsiAdv(PsiUp, PsiDn, B, theta0, psi0, X, spct = [1,1,1], Paxis = 0):
    #(theta,psi,fftup, ifftup, aup, faup, spct = 1,wavelength = 6e-8, sprd = 5e-3, dx=14.7E-7):
    
    '''
    Returns 3D wave function in sample, given the 3D field and incident angle
    
    # B - 3D internal field (Gauss)
    # theta - sample angle, deg
    # psi - sample angle, deg
    # fftup - 2D FFTW object
    # ifftup - 2D inverse FFTW object
    # aup - real space wave function object
    # faup - Fourier space wave function object
    
    # spct - optional argument to give z-direction aspect ratio of voxel
    # wavelength - neutron wavelength
    # sprd - angular spread (1 sigma) of beam in radians
    # dx - horizontal voxel dimension, cm
    '''
    
    if Paxis == 0:
        
        theta = 0. + theta0
        psi = 0. + psi0
        
    elif Paxis == 1:
        
        theta = np.sign(theta0) * 90 - theta0
        psi = 0. + psi0
        
    else:
        
        theta = 0. + theta0
        psi = np.sign(psi0) * 90 - psi0
        
    ### Compute Coefficient from constants ###
    ##########################################
    #mneutron=1.6749E-24#g
    #wavelength=6E-8#cm
    #hbar=1.0546E-27#cm^2gs^-1
    #wavelength = 10*wavelength
    #coeff = (1.91*5.05078E-24*dx*wavelength*mneutron)/(2*np.pi*(hbar**2))/np.cos(theta/180*np.pi)/np.cos(psi/180*np.pi)
    coeff = 2.312e6 * X.dx * X.wavelength / np.cos(theta/180*np.pi) / np.cos(psi/180*np.pi)
    
    dims = np.shape(B) #Dimension of input B field
   
    ### Declare Time evolution oerpator ###
    Operator = np.zeros([dims[1],dims[2],dims[3],2,2],dtype = np.csingle)
   
    
   
    ### Fill time evolution operator according to field ###
    Operator[:,:,:,0,0] = 1j*B[0]*coeff
    Operator[:,:,:,0,1] = (B[2]+1j*B[1])*coeff
    Operator[:,:,:,1,0] = (-B[2]+1j*B[1])*coeff
    Operator[:,:,:,1,1] = -1j*B[0]*coeff
    
    
    
    # Incident wave vector magnitude
    k0 = 2*np.pi/X.wavelength * X.dx # * np.cos(theta) * np.cos(psi)
    
    # Transverse wave vector
    if Paxis == 2:
        ky0 = k0*np.sin(theta/180*np.pi)
        kx0 = k0*np.sin(psi/180*np.pi)*np.cos(theta/180*np.pi)
    else:
        kx0 = k0*np.sin(theta/180*np.pi)
        ky0 = k0*np.sin(psi/180*np.pi)*np.cos(theta/180*np.pi)
    kz0 = (k0**2 - kx0**2 - ky0**2)**0.5
    
    if Paxis == 0:
        # declare q
        qx = np.linspace(0,dims[2]-1,dims[2]) - int((dims[2]+1)/2) 
        qy = np.linspace(0,dims[3]-1,dims[3]) - int((dims[3]+1)/2) 
    
        
        qx = np.roll(qx, -int(dims[2]/2))*np.pi*2/dims[2]/spct[1]
        qy = np.roll(qy, -int(dims[3]/2))*np.pi*2/dims[3]/spct[2]
        
        
        omg = np.full(PsiUp[-1].shape,0. + 0.j,dtype = np.csingle)
        omg += spct[0] *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5
        omg += - spct[0] *  kz0
    elif Paxis == 1:
        # declare q
        qx = np.linspace(0,dims[1]-1,dims[1]) - int((dims[1]+1)/2)
        qy = np.linspace(0,dims[3]-1,dims[3]) - int((dims[3]+1)/2) 
    
        
        qx = np.roll(qx, -int(dims[1]/2))*np.pi*2/dims[1]/spct[0]
        qy = np.roll(qy, -int(dims[3]/2))*np.pi*2/dims[3]/spct[2]
        
        
        omg = np.full(PsiUp[:,-1].shape,0. + 0.j,dtype = np.csingle)
        omg += spct[1] *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5
        omg += - spct[1] *  kz0
    else:
        # declare q
        qx = np.linspace(0,dims[1]-1,dims[1]) - int((dims[1]+1)/2)
        qy = np.linspace(0,dims[2]-1,dims[2]) - int((dims[2]+1)/2) 
    
        
        qx = np.roll(qx, -int(dims[1]/2))*np.pi*2/dims[1]/spct[0]
        qy = np.roll(qy, -int(dims[2]/2))*np.pi*2/dims[2]/spct[1]
        
        
        omg = np.full(PsiUp[:,:,-1].shape,0. + 0.j,dtype = np.csingle)
        omg += spct[2] *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5
        omg += - spct[2] *  kz0
    
    eomg = np.exp(1.j * omg) #np.sinc(omg) #* np.exp(1.j * omg/2) #* np.exp(-0.5 * omg**2 * sprd**2)
    eomg = eomg.astype(np.csingle)
    Operator = Operator.astype(np.csingle)
    if Paxis == 0:
    
        for i in range(dims[1]):
            
            # Real space wave from previous layer
            X.aup[Paxis][:] = PsiUp[i-1] #np.sum(Operator[i]*PsiUp[i-1,:,:,None,:],axis = 3)*spct
            # Momentum space wave
            X.faup[Paxis] = X.fftup[Paxis]()
            # Operator with free-space J(dz)
            X.faup[Paxis][:] *= eomg
            #faup[:] *= 1 / (np.sum(np.abs(faup)**2))**0.5
            # Real space wave function
            X.aup[Paxis] = X.ifftup[Paxis]()
            # Potential operates in real space
            #aup[:] += np.sum(Operator[i]*aup[:,:,None,:],axis = 3)*spct
            # Save in next layer
            PsiUp[i] = X.aup[Paxis] + np.sum(Operator[i]*X.aup[Paxis][:,:,None,:],axis = 3)*spct[0]
            
            # Repeat for incoming wave spin down
            X.aup[Paxis][:] = PsiDn[i-1] #np.sum(Operator[i]*PsiDn[i-1,:,:,None,:],axis = 3)*spct
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] *= eomg
            #faup[:] *= 1 / (np.sum(np.abs(faup)**2))**0.5
            X.aup[Paxis] = X.ifftup[Paxis]()
            #aup[:] += np.sum(Operator[i]*aup[:,:,None,:],axis = 3)*spct
            PsiDn[i] = X.aup[Paxis] + np.sum(Operator[i]*X.aup[Paxis][:,:,None,:],axis = 3)*spct[0]
            
    elif Paxis == 1:  #Add Paxis to new fftup aup list of fftw objects...
        
        for i in range(dims[2]):
            
            # Real space wave from previous layer
            X.aup[Paxis][:] = PsiUp[:,i-1] #np.sum(Operator[i]*PsiUp[i-1,:,:,None,:],axis = 3)*spct
            # Momentum space wave
            X.faup[Paxis] = X.fftup[Paxis]()
            # Operator with free-space J(dz)
            X.faup[Paxis][:] *= eomg
            #faup[:] *= 1 / (np.sum(np.abs(faup)**2))**0.5
            # Real space wave function
            X.aup[Paxis] = X.ifftup[Paxis]()
            # Potential operates in real space
            #aup[:] += np.sum(Operator[i]*aup[:,:,None,:],axis = 3)*spct
            # Save in next layer
            PsiUp[:,i] = X.aup[Paxis] + np.sum(Operator[:,i]*X.aup[Paxis][:,:,None,:],axis = 3) *spct[1]
            
            # Repeat for incoming wave spin down
            X.aup[Paxis][:] = PsiDn[:,i-1] #np.sum(Operator[i]*PsiDn[i-1,:,:,None,:],axis = 3)*spct
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] *= eomg
            #faup[:] *= 1 / (np.sum(np.abs(faup)**2))**0.5
            X.aup[Paxis] = X.ifftup[Paxis]()
            #aup[:] += np.sum(Operator[i]*aup[:,:,None,:],axis = 3)*spct
            PsiDn[:,i] = X.aup[Paxis] + np.sum(Operator[:,i]*X.aup[Paxis][:,:,None,:],axis = 3) *spct[1]
    
    else:
        
        for i in range(dims[3]):
            
            # Real space wave from previous layer
            X.aup[Paxis][:] = PsiUp[:,:,i-1] #np.sum(Operator[i]*PsiUp[i-1,:,:,None,:],axis = 3)*spct
            # Momentum space wave
            X.faup[Paxis] = X.fftup[Paxis]()
            # Operator with free-space J(dz)
            X.faup[Paxis][:] *= eomg
            #faup[:] *= 1 / (np.sum(np.abs(faup)**2))**0.5
            # Real space wave function
            X.aup[Paxis] = X.ifftup[Paxis]()
            # Potential operates in real space
            #aup[:] += np.sum(Operator[i]*aup[:,:,None,:],axis = 3)*spct
            # Save in next layer
            PsiUp[:,:,i] = X.aup[Paxis] + np.sum(Operator[:,:,i]*X.aup[Paxis][:,:,None,:],axis = 3) *spct[2]
            
            # Repeat for incoming wave spin down
            X.aup[Paxis][:] = PsiDn[:,:,i-1] #np.sum(Operator[i]*PsiDn[i-1,:,:,None,:],axis = 3)*spct
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] *= eomg
            #faup[:] *= 1 / (np.sum(np.abs(faup)**2))**0.5
            X.aup[Paxis] = X.ifftup[Paxis]()
            #aup[:] += np.sum(Operator[i]*aup[:,:,None,:],axis = 3)*spct
            PsiDn[:,:,i] = X.aup[Paxis] + np.sum(Operator[:,:,i]*X.aup[Paxis][:,:,None,:],axis = 3) *spct[2]
        
            
            
        


def ChiRev(ChiUp, ChiDn, B, theta0, psi0, X, spct = [1,1,1], Paxis = 0):
    #fftup, ifftup, aup, faup, spct = 1,wavelength = 6e-8, sprd = 5e-3,dx=14.7E-7):
   
    '''
    Propagates wave backwards through the sample, returns real space wave
    
    # B - 3D internal field (Gauss)
    # theta - sample angle, deg
    # psi - sample angle, deg
    # fftup - 2D FFTW object
    # ifftup - 2D inverse FFTW object
    
        - Note that these are switched, given reverse propagation
    # aup - FOURIER space wave function object 
    # faup - REAL space wave function object
        
    # Chi0Up - Final spinor leaving sample, given incident wave spin up
    # Chi0Dn - Final spinor leaving sample, given incident wave spin down
    
    # spct - optional argument to give z-direction aspect ratio of voxel
    # wavelength - neutron wavelength
    # sprd - angular spread (1 sigma) of beam in radians
    # dx - horizontal voxel dimension, cm
    '''
    
    
    if Paxis == 0:
        
        theta = 0. + theta0
        psi = 0. + psi0
        
    elif Paxis == 1:
        
        theta = np.sign(theta0) * 90 - theta0
        psi = 0. + psi0
        
    else:
        
        theta = 0. + theta0
        psi = np.sign(psi0) * 90 - psi0

    ### Compute Coefficient from constants ###
    ##########################################
    
    coeff = 2.312e6 * X.dx * X.wavelength / np.cos(theta/180*np.pi) / np.cos(psi/180*np.pi)
    
    dims = np.shape(B) #Dimension of input B field
   
    ### Declare Time evolution oerpator ###
    Operator = np.zeros([dims[1],dims[2],dims[3],2,2],dtype = np.csingle)
   
    
   
    ### Fill time evolution operator according to field ###
    Operator[:,:,:,0,0] = 1j*B[0]*coeff
    Operator[:,:,:,0,1] = (B[2]+1j*B[1])*coeff
    Operator[:,:,:,1,0] = (-B[2]+1j*B[1])*coeff
    Operator[:,:,:,1,1] = -1j*B[0]*coeff
    
   
    k0 = 2*np.pi/X.wavelength * X.dx
   
    #kx0 = k0*np.sin(theta/180*np.pi)
    #ky0 = k0*np.sin(psi/180*np.pi)*np.cos(theta/180*np.pi)
    
    if Paxis == 2:
        ky0 = k0*np.sin(theta/180*np.pi)
        kx0 = k0*np.sin(psi/180*np.pi)*np.cos(theta/180*np.pi)
    else:
        kx0 = k0*np.sin(theta/180*np.pi)
        ky0 = k0*np.sin(psi/180*np.pi)*np.cos(theta/180*np.pi)
    
    
    kz0 = (k0**2 - kx0**2 - ky0**2)**0.5
    
    
    if Paxis == 0:
        # declare q
        qx = np.linspace(0,dims[2]-1,dims[2]) - int((dims[2]+1)/2)
        qy = np.linspace(0,dims[3]-1,dims[3]) - int((dims[3]+1)/2) 
    
        
        qx = np.roll(qx, -int(dims[2]/2))*np.pi*2/dims[2]/spct[1]
        qy = np.roll(qy, -int(dims[3]/2))*np.pi*2/dims[3]/spct[2]
        
        
        omg = np.full(ChiUp[-1].shape,0. + 0.j,dtype = np.csingle)
        omg += spct[0] *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5
        omg += - spct[0] *  kz0
    elif Paxis == 1:
        # declare q
        qx = np.linspace(0,dims[1]-1,dims[1]) - int((dims[1]+1)/2)
        qy = np.linspace(0,dims[3]-1,dims[3]) - int((dims[3]+1)/2) 
    
        
        qx = np.roll(qx, -int(dims[1]/2))*np.pi*2/dims[1]/spct[0]
        qy = np.roll(qy, -int(dims[3]/2))*np.pi*2/dims[3]/spct[2]
        
        
        omg = np.full(ChiUp[:,-1].shape,0. + 0.j,dtype = np.csingle)
        omg += spct[1] *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5
        omg += - spct[1] *  kz0
    else:
        # declare q
        qx = np.linspace(0,dims[1]-1,dims[1]) - int((dims[1]+1)/2)
        qy = np.linspace(0,dims[2]-1,dims[2]) - int((dims[2]+1)/2) 
    
        
        qx = np.roll(qx, -int(dims[1]/2))*np.pi*2/dims[1]/spct[0]
        qy = np.roll(qy, -int(dims[2]/2))*np.pi*2/dims[2]/spct[1]
        
        
        omg = np.full(ChiUp[:,:,-1].shape,0. + 0.j,dtype = np.csingle)
        omg += spct[2] *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5
        omg += - spct[2] *  kz0
        
    '''
    qx = np.linspace(0,dims[2]-1,dims[2]) - int((dims[2]+1)/2)
    qy = np.linspace(0,dims[3]-1,dims[3]) - int((dims[3]+1)/2) 
    
    qx = np.roll(qx, -int(dims[2]/2))*np.pi*2/dims[2]
    qy = np.roll(qy, -int(dims[3]/2))*np.pi*2/dims[3]
    
    omg = np.full(ChiUp[-1].shape,0. + 0.j)
    omg += spct *  (k0**2 - (qx[:,None,None]+kx0)**2 - (qy[None,:,None]+ky0)**2 )**0.5   
    omg += - spct * kz0
    '''
    eomg = np.exp(1.j*omg) # * np.sinc(omg/2) * np.exp(-0.5 * omg**2 * sprd**2)
    
    eomg = eomg.astype(np.csingle)
    Operator = Operator.astype(np.csingle)
    
    if Paxis == 0:
    
        for i in range(dims[1] - 1):
            
            X.faup[Paxis][:] = ChiUp[-i-1] 
            X.aup[Paxis] = X.ifftup[Paxis]()
            X.aup[Paxis][:] *= eomg
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] += np.sum(Operator[-1-i]*X.faup[Paxis][:,:,:,None],axis = 2)*spct[0]
            ChiUp[-i-2] = X.faup[Paxis]
            
            X.faup[Paxis][:] = ChiDn[-i-1] 
            X.aup[Paxis] = X.ifftup[Paxis]()
            X.aup[Paxis][:] *= eomg
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] += np.sum(Operator[-1-i]*X.faup[Paxis][:,:,:,None],axis = 2)*spct[0]
            ChiDn[-i-2] = X.faup[Paxis]
            
    elif Paxis == 1:
        
        for i in range(dims[2] - 1):
            
            X.faup[Paxis][:] = ChiUp[:,-i-1] 
            X.aup[Paxis] = X.ifftup[Paxis]()
            X.aup[Paxis][:] *= eomg
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] += np.sum(Operator[:,-1-i]*X.faup[Paxis][:,:,:,None],axis = 2)*spct[1]
            ChiUp[:,-i-2] = X.faup[Paxis]
            
            X.faup[Paxis][:] = ChiDn[:,-i-1] 
            X.aup[Paxis] = X.ifftup[Paxis]()
            X.aup[Paxis][:] *= eomg
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] += np.sum(Operator[:,-1-i]*X.faup[Paxis][:,:,:,None],axis = 2)*spct[1]
            ChiDn[:,-i-2] = X.faup[Paxis]
        
    else:
        
        for i in range(dims[3] - 1):
            
            X.faup[Paxis][:] = ChiUp[:,:,-i-1] 
            X.aup[Paxis] = X.ifftup[Paxis]()
            X.aup[Paxis][:] *= eomg
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] += np.sum(Operator[:,:,-1-i]*X.faup[Paxis][:,:,:,None],axis = 2)*spct[2]
            ChiUp[:,:,-i-2] = X.faup[Paxis]
            
            X.faup[Paxis][:] = ChiDn[:,:,-i-1] 
            X.aup[Paxis] = X.ifftup[Paxis]()
            X.aup[Paxis][:] *= eomg
            X.faup[Paxis] = X.fftup[Paxis]()
            X.faup[Paxis][:] += np.sum(Operator[:,:,-1-i]*X.faup[Paxis][:,:,:,None],axis = 2)*spct[2]
            ChiDn[:,:,-i-2] = X.faup[Paxis]
        
        
    #return ChiUp, ChiDn, coeff


    
    
# =============================================================================
# Compute B given m
# =============================================================================

def BfromM(m, q, X):
   
    '''
    Returns b given m
    B still needs to be multiplied by saturated magnetization
    No guide field added
    
    # n3D - three-dimesional spin density Nz x Nx x Ny x 3
    # q - unit vector
    # fftm - fftW forward transform object
    # ifftm - fftW inverse transform object
    # a - real space vector field (starts same as n3D)
    # fa - Fourier space vector field
   
   
    ######## Outputs ##########
    ###########################  
    # B - three-dimesional spin density Nz x Nx x Ny x 3
    # Magnetic field with divergences removed, same dimensions as n3D
    '''
    
    # Fill realspace FFTW object argument with m
    X.a[:] = m
    
    # 3D Fourier Transform
    X.fa = X.fftm()
    
    # (q q).dot(fa)
    X.fa[:] = q * np.sum(X.fa*q,axis = 0)
    
    # Return to real space
    X.a = X.ifftm()
    
    return m - np.real(X.a)

def MakeQ(dims,spct = 1,rtn = 0):
    
    '''
    Returns 3D Q-hat vector coordinate
    
    # dims - dimensions of 3D array
    # spct - zeroth (z) dimension given an optional aspect argument
    '''
   
    q = np.zeros(dims)

    ### Construct filters for computing derivatives ###
    # +++++++++++++++++++++++++++++++++++++++++++++++ #
    qz = np.linspace(0,dims[1]-1,dims[1]) - dims[1]/2
    qx = np.linspace(0,dims[2]-1,dims[2]) - dims[2]/2
    qy = np.linspace(0,dims[3]-1,dims[3]) - dims[3]/2
   
    ### Roll filters to zero frequency at zero index ###
    qz = np.roll(qz, -int(dims[1]/2))/spct[0]
    qx = np.roll(qx, -int(dims[2]/2))/spct[1]
    qy = np.roll(qy, -int(dims[3]/2))/spct[2]
   
    #qnrm = np.full(dims,0.)  #declare q-normalization array
    #Fill qnrm array:
    qnrm = ( (qz[:,None,None])**2 + (qx[None,:,None] )**2 + (qy[None,None,:])**2 ) ** 0.5
    qnrm = np.maximum(1./np.max(spct),qnrm)
    
    q[0] = qz[:,None,None] / qnrm
    q[1] = qx[None,:,None] / qnrm
    q[2] = qy[None,None,:] / qnrm
    
    if rtn == 0:
        
        return q
    
    else:
        
        return q, qx, qy, qz

def MakeE(dims, theta,psi, spct = 1, wavelength = 6e-8, dx = 14.7e-7):
    
    
    #Incident wavevector magnitude
    k0 = 2*np.pi/wavelength * dx
    
    #Incident wavevector transverse components
    k0x = 2*np.pi*np.sin(theta)/wavelength*dx#/dims[1]*kx
    k0y = 2*np.pi*np.sin(psi)*np.cos(theta)/wavelength*dx#/dims[2]*ky
    
    k0z = np.sqrt(k0**2 - k0x**2 - k0y**2)
    '''
    #Omega(q) - wave z-component
    Stqx = np.linspace(0,dims[0]-1,dims[0]) - int((dims[0]+1)/2)
    Stqy = np.linspace(0,dims[1]-1,dims[1]) - int((dims[1]+1)/2) 
    
    Stqx = np.roll(Stqx, -int(dims[0]/2)) #*np.pi*2/dims[0]
    Stqy = np.roll(Stqy, -int(dims[1]/2)) #*np.pi*2/dims[1]
    
    Stomega = ((k0**2  -  (Stqx[:,None]*np.pi*2/dims[1]+k0x)**2 - (Stqy[None,:]*np.pi*2/dims[2]+k0y)**2 )**0.5 )* (dims[1]*dims[0])**0.5/(np.pi*2)
    Stomega += - Stomega[0,0]
    omq = Stomega.flatten()
    
    # q wavevectors
    qx = (Stqx[:,None] + 0.*Stqy[None,:]).flatten()
    qy = (0*Stqx[:,None] + Stqy[None,:]).flatten()
    '''
    
    
    
    
    Nxyz = dims[0]*dims[1]*dims[2]
    Nxy = dims[1]*dims[2]
    Nz = dims[0]
    Nx = dims[1]
    Ny = dims[2]
    
    qx = np.linspace(0,dims[1]-1,dims[1]) - dims[1]/2
    qy = np.linspace(0,dims[2]-1,dims[2]) - dims[2]/2
    
    qx = np.roll(qx, -int(dims[1]/2))
    qy = np.roll(qy, -int(dims[2]/2))
    
    qxp = (np.cos(theta) * qx[:,None] - np.sin(theta) * np.sin(psi) * qy[None,:]).flatten()
    qyp = ( np.cos(psi) * qy[None,:] + 0. * qx[:,None] ).flatten()
    qz = ((k0**2 - (k0x + qxp*np.pi*2/dims[1])**2 - (k0y + qyp*np.pi*2/dims[2])**2)**0.5 - k0z)/spct/(np.pi*2)*np.sqrt(dims[1]*dims[2]) #(np.sin(theta)*qx[:,None] + np.cos(theta) * np.sin(psi) * qy[None,:]).flatten()/spct
    
    
    fx = np.floor(qxp)
    fy = np.floor(qyp)
    fz = np.floor(qz)
    
    
    
    
    XW, YW = np.full((2,Nxy*8),0,dtype = np.int32)
    VW = np.full(Nxy*8, 0., dtype = np.float32)
    
    
    YW[:Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[:Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[:Nxy] = (fx + 1 - qxp)*(fy + 1 - qyp)*(fz + 1 - qz) 
    
    YW[Nxy:2*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[Nxy:2*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[Nxy:2*Nxy] = (fx + 1 - qxp)*(-fy + qyp)*(fz + 1 - qz) 
    
    YW[2*Nxy:3*Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[2*Nxy:3*Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[2*Nxy:3*Nxy] = (-fx + qxp)*(fy + 1 - qyp)*(fz + 1 - qz) 
    
    YW[3*Nxy:4*Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[3*Nxy:4*Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[3*Nxy:4*Nxy] = (fx + 1 - qxp)*(fy + 1 - qyp)*(- fz + qz) 
    
    YW[4*Nxy:5*Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[4*Nxy:5*Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[4*Nxy:5*Nxy] = (- fx + qxp)*(fy + 1 - qyp)*(- fz + qz)
    
    YW[5*Nxy:6*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[5*Nxy:6*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[5*Nxy:6*Nxy] = (fx + 1 - qxp)*(-fy + qyp)*(- fz + qz)
    
    YW[6*Nxy:7*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[6*Nxy:7*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[6*Nxy:7*Nxy] = (- fx + qxp)*(- fy + qyp)*( fz + 1 - qz)
    
    YW[7*Nxy:8*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[7*Nxy:8*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[7*Nxy:8*Nxy] = (- fx + qxp)*(- fy + qyp)*(- fz + qz)
    
    
    
    Wcoo = coo_matrix((VW,(YW,XW)), shape = (int(Nxy),int(Nxyz)))
    W = csr_matrix(Wcoo)
    
    return W

def MakeZ(dims, theta,psi, spct = 1):
    
    Nxyz = dims[0]*dims[1]*dims[2]
    Nxy = dims[1]*dims[2]
    Nz = dims[0]
    Nx = dims[1]
    Ny = dims[2]
    
    qx = np.linspace(0,dims[1]-1,dims[1]) - dims[1]/2
    qy = np.linspace(0,dims[2]-1,dims[2]) - dims[2]/2
    
    qx = np.roll(qx, -int(dims[1]/2))
    qy = np.roll(qy, -int(dims[2]/2))
    
    qxp = (np.cos(theta) * qx[:,None] - np.sin(theta) * np.sin(psi) * qy[None,:]).flatten()
    qyp = ( np.cos(psi) * qy[None,:] + 0. * qx[:,None] ).flatten()
    qz = (np.sin(theta)*qx[:,None] + np.cos(theta) * np.sin(psi) * qy[None,:]).flatten()/spct
    
    
    fx = np.floor(qxp)
    fy = np.floor(qyp)
    fz = np.floor(qz)
    
    
    
    
    XW, YW = np.full((2,Nxy*8),0,dtype = np.int32)
    VW = np.full(Nxy*8, 0., dtype = np.float32)
    
    
    YW[:Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[:Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[:Nxy] = (fx + 1 - qxp)*(fy + 1 - qyp)*(fz + 1 - qz) 
    
    YW[Nxy:2*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[Nxy:2*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[Nxy:2*Nxy] = (fx + 1 - qxp)*(-fy + qyp)*(fz + 1 - qz) 
    
    YW[2*Nxy:3*Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[2*Nxy:3*Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[2*Nxy:3*Nxy] = (-fx + qxp)*(fy + 1 - qyp)*(fz + 1 - qz) 
    
    YW[3*Nxy:4*Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[3*Nxy:4*Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[3*Nxy:4*Nxy] = (fx + 1 - qxp)*(fy + 1 - qyp)*(- fz + qz) 
    
    YW[4*Nxy:5*Nxy] = np.arange(Nxy) #np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[4*Nxy:5*Nxy] = np.mod(fy,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[4*Nxy:5*Nxy] = (- fx + qxp)*(fy + 1 - qyp)*(- fz + qz)
    
    YW[5*Nxy:6*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny
    XW[5*Nxy:6*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[5*Nxy:6*Nxy] = (fx + 1 - qxp)*(-fy + qyp)*(- fz + qz)
    
    YW[6*Nxy:7*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[6*Nxy:7*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz,Nz).astype(int)*Nx*Ny
    VW[6*Nxy:7*Nxy] = (- fx + qxp)*(- fy + qyp)*( fz + 1 - qz)
    
    YW[7*Nxy:8*Nxy] = np.arange(Nxy) #np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny
    XW[7*Nxy:8*Nxy] = np.mod(fy+1,Ny).astype(int) + np.mod(fx+1,Nx).astype(int)*Ny + np.mod(fz+1,Nz).astype(int)*Nx*Ny
    VW[7*Nxy:8*Nxy] = (- fx + qxp)*(- fy + qyp)*(- fz + qz)
    
    
    
    Wcoo = coo_matrix((VW,(YW,XW)), shape = (int(Nxy),int(Nxyz)))
    W = csr_matrix(Wcoo)
    
    return W

def Zproj(dims,Projns,ax = 1,spct = 1):
    
    
    '''
    Returns two lists, of W and WT, both is sparse row format
    
    # dims - dimesions of diffraction pattern
    # Projns - Sample rotation angles
    # axis == 0 - Rotation about theta
      axis == 1 - Rotation about psi
      axis = array Rotation about psi, for each Projns rotation about theta
    '''
    
    Ws = []
    #WsT = []
    
    if isinstance(ax, int) and ax == 0:
        
        for i in range(Projns.shape[0]):
            W = MakeZ(dims,Projns[i], 0., spct = spct)
            Ws.append(W)
            #WsT.append(WT)
        
    elif isinstance(ax, int) and ax == 1:
        
        for i in range(Projns.shape[0]):
            W = MakeZ(dims,0.,Projns[i],spct = spct)
            Ws.append(W)
            #WsT.append(WT)
    else:
        
        for i in range(Projns.shape[0]):
            W, WT = MakeZ(dims,Projns[i],ax[i],spct = spct)
            Ws.append(W)
            #WsT.append(WT)
    
    return Ws

def Eproj(dims,Projns,ax = 1,spct = 1):
    
    
    '''
    Returns two lists, of W and WT, both is sparse row format
    
    # dims - dimesions of diffraction pattern
    # Projns - Sample rotation angles
    # axis == 0 - Rotation about theta
      axis == 1 - Rotation about psi
      axis = array Rotation about psi, for each Projns rotation about theta
    '''
    
    Ws = []
    #WsT = []
    
    if isinstance(ax, int) and ax == 0:
        
        for i in range(Projns.shape[0]):
            W = MakeE(dims,Projns[i], 0., spct = spct)
            Ws.append(W)
            #WsT.append(WT)
        
    elif isinstance(ax, int) and ax == 1:
        
        for i in range(Projns.shape[0]):
            W = MakeE(dims,0.,Projns[i],spct = spct)
            Ws.append(W)
            #WsT.append(WT)
    else:
        
        for i in range(Projns.shape[0]):
            W, WT = MakeE(dims,Projns[i],ax[i],spct = spct)
            Ws.append(W)
            #WsT.append(WT)
    
    return Ws

def MakeY3D(dims,Fd,xFd):
    
    Y = MakeY(dims[2:4],Fd,xFd)
    
    Y += MakeY((dims[1],dims[3]),Fd,xFd)
    
    Y += MakeY((dims[1],dims[2]),Fd,xFd)
    
    return Y[0::2], Y[1::2]

def MakeW3D(shape, ProjTh, ProjPsi, dimscatt = None):
    
    W = Wproj(shape[2:4], ProjTh, ProjPsi, dimscatt, 0)
    
    W += Wproj((shape[1],shape[3]), ProjTh, ProjPsi, dimscatt, 1)
    
    W += Wproj((shape[1],shape[2]), ProjTh, ProjPsi, dimscatt, 2)
    
    return W[0::2], W[1::2]
    
    

# =============================================================================
# Transformation Given Wavelength Spread
# =============================================================================

def MakeY(dims,Fd,xFd):
    
    ''' 
    Return Csr Sparse matrix which operates on a [0,0] - centered scattering 
    pattern
    
    dims - dimensions of scattering pattern
    
    Fd and and xFd Fd is the dk / k0 wavelength distribution (xFd in radians) 
    
    W.dot(A.flatten()).reshape(shape(A)) 
    '''
    
    #dims = np.shape(Suu)
    
    Stqx = np.linspace(0,dims[0]-1,dims[0]) - int((dims[0]+1)/2)
    Stqy = np.linspace(0,dims[1]-1,dims[1]) - int((dims[1]+1)/2) 
    
    Stqx = np.roll(Stqx, -int(dims[0]/2)) #*np.pi*2/dims[0]
    Stqy = np.roll(Stqy, -int(dims[1]/2)) #*np.pi*2/dims[1]
    
    qx = (Stqx[:,None] + 0.*Stqy[None,:]).flatten()
    qy = (0*Stqx[:,None] + Stqy[None,:]).flatten()
    
    Nx = dims[0]
    Ny = dims[1]
    Nxy = int(dims[0]*dims[1])
    Nd = Fd.shape[0]
    
    XW = np.full(Nxy*4*Nd,0,dtype = np.int32)
    YW = np.full(Nxy*4*Nd,0,dtype = np.int32)
    VW = np.full(Nxy*4*Nd,0,dtype = np.float32)
    
    for i in range(Nd):
        
        st = i*4*Nxy
        
        qxp = qx*(1+xFd[i])
        qyp = qy * (1+xFd[i])
        fqxp = np.floor(qxp)
        fqyp = np.floor(qyp)
        
        YW[st:Nxy+st] = np.arange(Nxy)
        XW[st:Nxy+st] = np.mod(fqyp,Ny).astype(int) + np.mod(fqxp,Nx).astype(int)*Nx
        VW[st:Nxy+st] = Fd[i]*(fqxp + 1 - qxp)*(fqyp + 1 - qyp)
        
        YW[Nxy+st:2*Nxy+st] = np.arange(Nxy)
        XW[Nxy+st:2*Nxy+st] = np.mod(fqyp+1,Ny).astype(int) + np.mod(fqxp,Nx).astype(int)*Nx
        VW[Nxy+st:2*Nxy+st] = Fd[i]*(fqxp + 1 - qxp)*(qyp - fqyp)
        
        YW[2*Nxy+st:3*Nxy+st] = np.arange(Nxy)
        XW[2*Nxy+st:3*Nxy+st] = np.mod(fqyp,Ny).astype(int) + np.mod(fqxp+1,Nx).astype(int)*Nx
        VW[2*Nxy+st:3*Nxy+st] = Fd[i]*(qxp - fqxp)*(1 + fqyp - qyp)
        
        YW[3*Nxy+st:4*Nxy+st] = np.arange(Nxy)
        XW[3*Nxy+st:4*Nxy+st] = np.mod(fqyp+1,Ny).astype(int) + np.mod(fqxp+1,Nx).astype(int)*Nx
        VW[3*Nxy+st:4*Nxy+st] = Fd[i]*(qxp - fqxp)*(qyp - fqyp)

    Wcoo = coo_matrix((VW,(XW,YW)), shape = (int(Nxy),int(Nxy)))
    W = csr_matrix(Wcoo)
    WT = W.transpose().tocsr()
    return W, WT


# =============================================================================
# Projection operator for outgoing wave
# =============================================================================

def MakeWS(shape,Theta0,Psi0,dimscatt = None, Paxis = 0, spct = [1,1,1]):
    
    '''
    Returns matrix that transforms scattering pattern to projection
    perpendicular to to incident wave
    
    # shape - dimensions of m
    # Theta - Rotation of sample, deg
    # Psi - rotation of sample, det
    # dimscatt - dimensions of scattering pattern if different from m-slice
    
    Optional:
    # Wavelength - neutron wavelength, cm
    # dx - pixel size, cm
    '''
    
        
    if len(shape) == 2:
        dims = shape
        Theta = 0 + Theta0
        Psi = 0 + Psi0
    else:
        if Paxis == 0:
            dims = shape[2:4]
            Theta = 0 + Theta0
            Psi = 0 + Psi0
            Qspct = 1 / np.array(spct[1:])
        elif Paxis == 1:
            dims = (shape[1],shape[3])
            Theta = - Theta0 + np.sign(Theta0) * 90
            Psi = 0 + Psi0
            Qspct = 1 / np.array(spct[::2])
        else:
            dims = (shape[1],shape[2])
            Theta = 0 + Theta0
            Psi = - Psi0 + np.sign(Psi0) * 90
            Qspct = 1 / np.array(spct[:-1])
            
    if dimscatt == None:
        dimscatt = dims
    
    Nx = dimscatt[0]
    Ny = dimscatt[1]
    Nxy = int(dimscatt[0]*dimscatt[1])
    
    
    Stqx = np.linspace(0,Nx-1,Nx) - int((Nx+1)/2)
    Stqy = np.linspace(0,Ny-1,Ny) - int((Ny+1)/2) 
    
    Stqx = np.roll(Stqx, -int(Nx/2)) #*np.pi*2/dims[0]
    Stqy = np.roll(Stqy, -int(Ny/2)) #*np.pi*2/dims[1]
    
    qx = (Stqx[:,None] + 0.*Stqy[None,:]).flatten()
    qy = (0*Stqx[:,None] + Stqy[None,:]).flatten()
    
    
    # Declare matrix adresses and values
    XW = np.full(Nxy*4,0,dtype = np.int32) # X Location
    YW = np.full(Nxy*4,0,dtype = np.int32) # Y location
    VW = np.full(Nxy*4,0,dtype = np.float32) # Matrix Values
    
    # Argument of tranformation delta function
    qxp = qx * np.cos(Theta/180*np.pi) #+ np.sin(Theta/180*np.pi) * omq #-  np.sin(Theta/180*np.pi) * ( np.sin(Psi/180*np.pi) * qy + np.cos(Psi/180*np.pi) * omq )
    qyp = qy * np.cos(Psi/180*np.pi) + np.sin(Psi/180*np.pi) * np.sin(Theta/180*np.pi) * qx #+ np.sin(Psi/180*np.pi) *(np.sin(Theta/180*np.pi)*qx + np.cos(Theta/180*np.pi) * omq) #omq * np.sin(Psi/180*np.pi)
    
    #qxp = qx / np.cos(Theta/180*np.pi)
    #qyp = (qy + np.sin(Psi/180*np.pi) * np.tan(Theta/180*np.pi))/np.cos(Psi/180*np.pi)
    
    # Floor of qxp, qyp
    fqxp = np.floor(qxp)
    fqyp = np.floor(qyp)
    
    # Each qxp, qpy subpixel resolution gives four adjacent points
    YW[:Nxy] = np.arange(Nxy)
    XW[:Nxy] = np.mod(fqyp,Nx).astype(int) + np.mod(fqxp,Ny).astype(int)*Ny
    VW[:Nxy] = (fqxp + 1 - qxp)*(fqyp + 1 - qyp)
    
    YW[Nxy:2*Nxy] = np.arange(Nxy)
    XW[Nxy:2*Nxy] = np.mod(fqyp+1,Nx).astype(int) + np.mod(fqxp,Ny).astype(int)*Ny
    VW[Nxy:2*Nxy] = (fqxp + 1 - qxp)*(qyp - fqyp)
    
    YW[2*Nxy:3*Nxy] = np.arange(Nxy)
    XW[2*Nxy:3*Nxy] = np.mod(fqyp,Nx).astype(int) + np.mod(fqxp+1,Ny).astype(int)*Ny
    VW[2*Nxy:3*Nxy] = (qxp - fqxp)*(1 + fqyp - qyp)
    
    YW[3*Nxy:4*Nxy] = np.arange(Nxy)
    XW[3*Nxy:4*Nxy] = np.mod(fqyp+1,Nx).astype(int) + np.mod(fqxp+1,Ny).astype(int)*Ny
    VW[3*Nxy:4*Nxy] = (qxp - fqxp)*(qyp - fqyp)
    
    # Make Matrix from location and value arrays
    Wcoo = coo_matrix((VW,(YW,XW)), shape = (int(Nxy),int(Nxy)))
    W = csr_matrix(Wcoo)
    WT = W.transpose().tocsr()
    
    if Paxis == 0:
        
        condt = spct[1] != spct[2]
    
    elif Paxis == 1:
        
        condt = spct[0] != spct[2]
        
    else:
        
        condt = spct[0] != spct[1]
    
    if dims != dimscatt or condt:
        
        Nx2 = dims[0]
        Ny2 = dims[1]
        Nxy2 = int(Nx2 * Ny2)
        
        Stqx = np.linspace(0,Nx2-1,Nx2) - int((Nx2+1)/2)
        Stqy = np.linspace(0,Ny2-1,Ny2) - int((Ny2+1)/2) 
        
        Stqx = np.roll(Stqx, -int(Nx2/2)) #*np.pi*2/dims[0]
        Stqy = np.roll(Stqy, -int(Ny2/2)) #*np.pi*2/dims[1]
        
        qx = (Stqx[:,None] + 0.*Stqy[None,:]).flatten()
        qy = (0*Stqx[:,None] + Stqy[None,:]).flatten()
        
        # Declare matrix adresses and values
        XW = np.full(Nxy2*4,0,dtype = np.int32) # X Location
        YW = np.full(Nxy2*4,0,dtype = np.int32) # Y location
        VW = np.full(Nxy2*4,0,dtype = np.float32) # Matrix Values
        
        # Argument of tranformation delta function
        #qxp = qx * np.cos(Theta/180*np.pi) #+ np.sin(Theta/180*np.pi) * omq #-  np.sin(Theta/180*np.pi) * ( np.sin(Psi/180*np.pi) * qy + np.cos(Psi/180*np.pi) * omq )
        #qyp = qy * np.cos(Psi/180*np.pi) + np.sin(Psi/180*np.pi) * np.sin(Theta/180*np.pi) * qx #+ np.sin(Psi/180*np.pi) *(np.sin(Theta/180*np.pi)*qx + np.cos(Theta/180*np.pi) * omq) #omq * np.sin(Psi/180*np.pi)
        
        #qxp = qx / np.cos(Theta/180*np.pi)
        #qyp = (qy + np.sin(Psi/180*np.pi) * np.tan(Theta/180*np.pi))/np.cos(Psi/180*np.pi)
        if Paxis == 2:
            qyp = np.sign(Psi0) * qx * Nx/Nx2 * Qspct[0]
            qxp = qy * Ny/Ny2 * Qspct[1]
        elif Paxis == 1:
            qxp = np.sign(Theta0) * qx * Nx/Nx2 * Qspct[0]
            qyp = qy * Ny/Ny2 * Qspct[1]
        else:
            qxp = qx * Nx/Nx2 * Qspct[0]
            qyp = qy * Ny/Ny2 * Qspct[1]
        
        # Floor of qxp, qyp
        fqxp = np.floor(qxp)
        fqyp = np.floor(qyp)
        
        # Each qxp, qpy subpixel resolution gives four adjacent points
        XW[:Nxy2] = np.arange(Nxy2)
        YW[:Nxy2] = np.mod(fqyp,Ny).astype(int) + np.mod(fqxp,Nx).astype(int)*Nx
        VW[:Nxy2] = (fqxp + 1 - qxp)*(fqyp + 1 - qyp)
        
        XW[Nxy2:2*Nxy2] = np.arange(Nxy2)
        YW[Nxy2:2*Nxy2] = np.mod(fqyp+1,Ny).astype(int) + np.mod(fqxp,Nx).astype(int)*Nx
        VW[Nxy2:2*Nxy2] = (fqxp + 1 - qxp)*(qyp - fqyp)
        
        XW[2*Nxy2:3*Nxy2] = np.arange(Nxy2)
        YW[2*Nxy2:3*Nxy2] = np.mod(fqyp,Ny).astype(int) + np.mod(fqxp+1,Nx).astype(int)*Nx
        VW[2*Nxy2:3*Nxy2] = (qxp - fqxp)*(1 + fqyp - qyp)
        
        XW[3*Nxy2:4*Nxy2] = np.arange(Nxy2)
        YW[3*Nxy2:4*Nxy2] = np.mod(fqyp+1,Ny).astype(int) + np.mod(fqxp+1,Nx).astype(int)*Nx
        VW[3*Nxy2:4*Nxy2] = (qxp - fqxp)*(qyp - fqyp)
        
        # Make Matrix from location and value arrays
        Zcoo = coo_matrix((VW,(YW,XW)), shape = (int(Nxy),int(Nxy2)))
        Z = csr_matrix(Zcoo)
        #ZT = W.transpose().tocsr()
    
        W = W.dot(Z)
        WT = W.transpose().tocsr()
    
    
    return W, WT

def Wproj(dims,Projns,ax = 0, dims2 = None, Paxis = 0):
    
    
    '''
    Returns two lists, of W and WT, both is sparse row format
    
    # dims - dimesions of m-slice
    # Projns - Sample rotation angles
    # axis == 0 - Rotation about theta
      axis == 1 - Rotation about psi
      axis = array Rotation about psi, for each Projns rotation about theta
      dims2 = dimensions of diffraction pattern if different from m-slice
    '''
    
    if dims2 == None:
        dims2 = dims
    
    Ws = []
    WsT = []
    
    if isinstance(ax, int) and ax == 0:
        
        for i in range(Projns.shape[0]):
            W, WT = MakeWS(dims,0.,Projns[i], dims2, Paxis)
            Ws.append(W)
            WsT.append(WT)
        
    elif isinstance(ax, int) and ax == 1:

        
        for i in range(Projns.shape[0]):
            W, WT = MakeWS(dims,Projns[i],0., dims2, Paxis)
            Ws.append(W)
            WsT.append(WT)
        
        
    else:
        
        for i in range(Projns.shape[0]):
            W, WT = MakeWS(dims,Projns[i],ax[i], dims2, Paxis)
            Ws.append(W)
            WsT.append(WT)
    
    return Ws, WsT



def WprojA(dims,Projns,ax = 0, dims2 = None, Paxis = np.array(None), spct = [1,1,1]):
    
    
    '''
    Returns two lists, of W and WT, both is sparse row format
    
    # dims - dimesions of m
    # Projns - Sample rotation angles
    # axis == 0 - Rotation about theta
      axis == 1 - Rotation about psi
      axis = array Rotation about psi, for each Projns rotation about theta
      dims2 = dimensions of diffraction pattern if different from m-slice
    '''
    
    if Paxis.all == None:
        Paxis = (0 * Projns).astype(int)
    
    if dims2 == None:
        dims2 = dims
    
    Ws = []
    WsT = []
    
    if isinstance(ax, int) and ax == 0:
        
        for i in range(Projns.shape[0]):
            W, WT = MakeWS(dims,0.,Projns[i], dims2, Paxis[i], spct)
            Ws.append(W)
            WsT.append(WT)
        
    elif isinstance(ax, int) and ax == 1:

        
        for i in range(Projns.shape[0]):
            W, WT = MakeWS(dims,Projns[i],0., dims2, Paxis[i], spct)
            Ws.append(W)
            WsT.append(WT)
        
        
    else:
        
        for i in range(Projns.shape[0]):
            W, WT = MakeWS(dims,Projns[i],ax[i], dims2, Paxis[i], spct)
            Ws.append(W)
            WsT.append(WT)
    
    return Ws, WsT





