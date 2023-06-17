# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:52:36 2022

@author: bjh3
"""

import numpy as np
from . import ChiSq as CS
from scipy import ndimage

class SpinDensity:
    '''
    '''
    
    def __init__(self, m, M0, H0, spct = [1,1,1], dx = 14.7e-7):
        self.m = m.astype(np.float32)
        self.M0 = M0 #.astype(np.float32)
        self.H0 = H0 #.astype(np.float32)
        self.voxelaspect = np.array(spct)
        self.qhat = CS.MakeQ(m.shape, spct = spct).astype(np.float32)
        self.dx = dx
        
    def initF(self, Q0, field):
        self.Q0 = Q0
        self.h = field
        self.beta = field * Q0**2 / 2.
        self.fieldhat = field / np.sum(field**2)**0.5
        self.dFdm = np.full(self.m.shape,0.,dtype=np.float32)
        self.Lm = np.full(self.m.shape,0.,dtype=np.float32)
        self.dm = np.full((3,) + self.m.shape,0.,dtype = np.float32)
        self.hHeis = np.full(self.m.shape[1:],0.,dtype = np.float32)
        self.hDM = np.full(self.m.shape[1:],0.,dtype = np.float32)
        self.Fden = np.full(self.m.shape[1:],0.,dtype = np.float32)

    def Grad(self):
        
        for j in range(3):
            self.dm[:,j,:] = 0.5 * (np.roll(self.m,-1,axis = j+1) - np.roll(self.m,1,axis = j+1))
            self.dm[:,j] *= 1/self.voxelaspect[j]
    
    def Lap(self):
        
        self.Lm *= 0.
        #for i in range(2):
        #    self.Lm += np.roll(self.m,2*i-1,axis=1) / self.voxelaspect**2
        for j in range(3):
            for i in range(2):
                self.Lm += np.roll(self.m,2*i-1,axis=j+1) / self.voxelaspect[j]**2
        self.Lm += - 2 * np.sum(1/self.voxelaspect**2) * self.m
        
    def ComputeFden(self):
        
        self.Lap()
        self.Grad()
        
        self.hHeis[:] = - 0.5 * np.sum(self.m * self.Lm,axis=0)
        
        self.hDM[:] = self.Q0 * (self.m[2] * (self.dm[1,0] - self.dm[0,1]) + 
                    self.m[0] * (self.dm[2,1] - self.dm[1,2]) 
                    - self.m[1] * (self.dm[2,0] - self.dm[0,2]))
        
        self.Fden[:] = self.hHeis + self.hDM - np.sum(self.beta[:,None,None,None] * self.m,axis=0)
        
        self.Ftot = np.sum(self.Fden)
        
    def FGrad(self):
        
        self.ComputeFden()
        
        self.dFdm = self.Lm
        self.dFdm[0] += -2 * self.Q0 * (self.dm[2,1] - self.dm[1,2])
        self.dFdm[1] +=  2 * self.Q0 * (self.dm[2,0] - self.dm[0,2])
        self.dFdm[2] += -2 * self.Q0 * (self.dm[1,0] - self.dm[0,1])
        
        self.dFdm += self.beta[:,None,None,None]
        
    def ComputeMetas(self):
        
        self.ComputeFden()
        self.emb = np.full(self.m.shape,0.,dtype = np.float32)
        self.emb[0] = (self.m[1] * (self.dm[2,1] * self.dm[0,2] - self.dm[2,2] * self.dm[0,1]) +
            + self.m[2] * (self.dm[0,1] * self.dm[1,2] - self.dm[0,0] * self.dm[0,2]) 
            + self.m[0] * (self.dm[1,1] * self.dm[2,2] - self.dm[1,2] * self.dm[2,1]))/(4*np.pi)
        self.emb[1] = (self.m[1] * (self.dm[1,1] * self.dm[0,0] - self.dm[2,0] * self.dm[0,2]) +
            + self.m[2] * (self.dm[0,2] * self.dm[1,0] - self.dm[1,2] * self.dm[0,0]) 
            + self.m[0] * (self.dm[1,2] * self.dm[2,0] - self.dm[1,0] * self.dm[2,2]))/(4*np.pi)
        self.emb[2] = (self.m[1] * (self.dm[2,0] * self.dm[0,1] - self.dm[2,1] * self.dm[0,0]) +
            + self.m[2] * (self.dm[0,0] * self.dm[1,1] - self.dm[1,0] * self.dm[0,1]) 
            + self.m[0] * (self.dm[1,0] * self.dm[2,1] - self.dm[1,1] * self.dm[2,0]))/(4*np.pi)
        self.rhoEm = np.full(self.m.shape[1:],0.,dtype = np.float32)
        self.rhoEm[:] += 0.5 * ((np.roll(self.emb[0],-1,axis = 0) - np.roll(self.emb[0],1,axis = 0))/self.voxelaspect[0]
                            + (np.roll(self.emb[1],-1,axis = 1) - np.roll(self.emb[1],1,axis = 1))/self.voxelaspect[1]
                            + (np.roll(self.emb[2],-1,axis = 2) - np.roll(self.emb[2],1,axis = 2))/self.voxelaspect[2])
        self.rhoM = np.full(self.m.shape[1:],0.,dtype = np.float32)
        self.rhoM[:] += 0.5 * ((np.roll(self.m[0],-1,axis = 0) - np.roll(self.m[0],1,axis = 0))/self.voxelaspect[0]
                            + (np.roll(self.m[1],-1,axis = 1) - np.roll(self.m[1],1,axis = 1))/self.voxelaspect[1]
                            + (np.roll(self.m[2],-1,axis = 2) - np.roll(self.m[2],1,axis = 2))/self.voxelaspect[2])
        self.rhoM = np.full(self.m.shape[1:],0.,dtype = np.float32)
        self.rhoM[:] += 0.5 * ((np.roll(self.m[0],-1,axis = 0) - np.roll(self.m[0],1,axis = 0))/self.voxelaspect[0]
                            + (np.roll(self.m[1],-1,axis = 1) - np.roll(self.m[1],1,axis = 1))/self.voxelaspect[1]
                            + (np.roll(self.m[2],-1,axis = 2) - np.roll(self.m[2],1,axis = 2))/self.voxelaspect[2])
        
    def Angular(self):
        
        self.TH = np.arctan2(np.sqrt(self.m[1]**2 + self.m[2]**2),self.m[0])
        self.PH = np.arctan2(self.m[2],self.m[1])
        
    def AngularGrad(self, gm):
        
        return gm[1] * np.cos(self.TH) * np.cos(self.PH) + \
            gm[2] * np.cos(self.TH) * np.sin(self.PH) \
            - gm[0] * np.sin(self.TH), \
                - gm[1] * self.m[2] + gm[2] * self.m[1]
    
    def CalcBetaAngular(self, dCostdm, hTH, hPH):
        
        beta = np.full(2,0.)
          
        beta[0] = np.sum(hTH * ( - dCostdm[0] * np.sin(self.TH) + 
                        dCostdm[1] * np.cos(self.TH) * np.cos(self.PH) +
                        dCostdm[2] * np.cos(self.TH) * np.sin(self.PH) ))
                        
        
        beta[1] = np.sum(hPH * ( - dCostdm[1] * np.sin(self.TH) * 
                                                            np.sin(self.PH) +
                            dCostdm[2] * np.sin(self.TH) * np.cos(self.PH)))
        
        return beta
    
    def ComputeB(self, X):
        
        self.B = CS.BfromM(self.m, self.qhat, X) * self.M0 + self.H0 * self.fieldhat[:,None,None,None]
        

    def MonopoleList(self, rng = 2, lvl = 0.02):
        
        pks = PkMx(self.rhoEm**2,lvl**2)
        self.locs = np.array(np.where(pks>0)).transpose()
        self.NMP = self.locs.shape[0]
        self.EmCh, self.dz = np.full((2,self.NMP),0.)
        self.DefectType = [None]*self.NMP
        self.NSp = 0
        self.NSm = 0
        self.NBp = 0
        self.NBm = 0
        
        for j in range(self.NMP):
            for z in range(self.locs[j,0]-rng,self.locs[j,0]+rng+1,1):
                for x in range(self.locs[j,1]-rng,self.locs[j,1]+rng+1,1):
                    for y in range(self.locs[j,2]-rng,self.locs[j,2]+rng+1,1):
                        
                        zi = np.mod(z,self.m.shape[1])
                        xi = np.mod(x,self.m.shape[2])
                        yi = np.mod(y,self.m.shape[3])
                        self.EmCh[j] += self.rhoEm[zi,xi,yi]
                        self.dz[j] += self.dm[0,0,zi,xi,yi]
                        
                        
            if self.EmCh[j] > 0 and self.dz[j] < 0:
                
                self.DefectType[j] = 'B+'
                self.NBp += 1
            
            elif self.EmCh[j] < 0 and self.dz[j] > 0:
                
                self.DefectType[j] = 'B-'
                self.NBm += 1
            
            elif self.EmCh[j] < 0 and self.dz[j] < 0:
                
                self.DefectType[j] = 'S-'
                self.NSm += 1
            
            else:
                
                self.DefectType[j] = 'S+'
                self.NSp += 1
                

class chi2:
    
    def __init__(self, shape, fIm, fI0, ProjTh, ProjPsi, N = np.array([1,1,1]), 
                 wts = np.array(None), wavelengthdist = ['triangular',0.1], 
                 wavelength = 6e-8, dx = 14.7e-7, 
                 Paxis = np.array(None), PsiSwitch = 45, ThetaSwitch = 45,
                 spct = [1,1,1]):
        
        self.mshape = shape
        #self.voxelaspect = spct
        
        self.fIm = fIm.astype(np.float32)
        self.fI0 = fI0.astype(np.float32)
        self.I0 = np.fft.fftn(fI0)
        self.ProjTh = ProjTh
        self.ProjPsi = ProjPsi
        self.N = N
        self.dx = dx
        
        if Paxis.all() == None:
            Paxis = (0 * ProjTh).astype(int)
            Paxis += np.sign(np.clip(np.abs(ProjTh) - ThetaSwitch,0,None)).astype(int)
            Paxis += 2 * np.sign(np.clip(np.abs(ProjPsi) - PsiSwitch,0,None)).astype(int)
        
        self.Paxis = Paxis
        
        self.dimscatt = (fI0.shape[-2],fI0.shape[-1])
        
        if wts.all() == None:
            self.weights = 1 / np.maximum(fIm,1) #Poisson weights
        else:
            self.weights = wts
        
        #self.fftm, self.ifftm, self.a, self.fa, self.fftup, self.ifftup, \
        #    self.aup, self.faup = CS.MakeFFTW(shape)
        
        self.fftm, self.ifftm, self.a, self.fa, self.fftup, self.ifftup, \
            self.aup, self.faup = CS.MakeFFTW3D(shape)
        
        if wavelengthdist[0] == 'gaussian':
            
            sprd = wavelengthdist[-1]
            
            xFd = np.linspace(-3*sprd,3*sprd,41)
            Fd = np.exp(-0.5 * xFd**2)
            Fd = Fd / np.sum(Fd)
            
        else: #triangular
        
            sprd = wavelengthdist[-1]
            
            xFd = np.linspace(-sprd,sprd,41)
            Fd = sprd - np.abs(xFd)
            Fd = Fd / np.sum(Fd)
            
        
        #self.Y, self.YT = CS.MakeY(shape[2:4], Fd, xFd)
        
        self.wavelength = wavelength #dist[1]
        
        #self.W, self.WT = CS.Wproj(shape[2:4],ProjTh, ax = ProjPsi, wavelength = wavelength)
        
        self.Y, self.YT = CS.MakeY(self.dimscatt, Fd, xFd)
        
        self.W, self.WT = CS.WprojA(shape, ProjTh, ProjPsi, self.dimscatt, Paxis)
        
        self.dchidm = np.full(shape,0.,dtype = np.float32)
        
        
    def ComputeGrad(self, S, ld = False):
        
        '''
        self.chisq, self.Ikt = CS.dChidM(self.dchidm, S.m, S.M0, S.H0, 
            
            self.fIm, self.ProjTh, self.ProjPsi, self.weights, self.I0, 
            self.N, S.qhat, 
            
            self.Y, self.YT, self.W, self.WT, 
            
            self.fftm, self.ifftm, self.a, self.fa, self.fftup, self.ifftup, 
            self.aup, self.faup, 
            
            spct = S.voxelaspect, wavelength = self.wavelength, ld = ld)
        '''
        
        self.chisq, self.Ikt = CS.dChidM(self, S, False)
        
    def ComputeResids(self, S):
        
        #self.Ikt, self.ss = CS.Fwrd(S.m, S.M0, S.H0, 
        #    self.N, self.ProjTh, self.ProjPsi, self.fI0, spct = S.voxelaspect)
        
        self.Ikt, self.ss = CS.Fwrd(self, S)
        
        self.chisq = np.sum((self.fIm - self.Ikt)**2 * self.weights)

        
def PkMx(A,mnv = 0,sz = 0.5, nh = 1):
    
    pks = np.full(A.shape,1,dtype=np.int)
    
    Ag = ndimage.gaussian_filter(A,sz,mode='wrap')
    
    
    strt = len(A.shape)-3
    axs = (strt,strt+1,strt+2)
    
    for i in range(-nh,nh+1):
        for j in range(-nh,nh+1):
            for k in range(-nh,nh+1):
                
                if i==j==k==0:
                    pass
                else:
                    pks *= np.sign(np.clip(Ag - np.roll(Ag, (i,j,k),axis=axs),0,None)).astype(int)
    
    pks *= np.sign(np.clip(Ag-mnv,0,None)).astype(int)
    
    return pks


def FwrdInitSX(m, M0, H0, N, ProjTh, ProjPsi, fI0, spct = 1, 
               wavelength = 6e-8, dx = 14.7e-7, ld = True,
               rtnSX = False):
    
    S = SpinDensity(m, M0, H0, spct, dx)
    X = CS.PropParams(m.shape, ProjTh, ProjPsi, fI0, wavelength, dx)
    
    if rtnSX:
        
        return S, X, CS.Fwrd(X, S, ld)
    
    else:
        
        return CS.Fwrd(X, S, ld)