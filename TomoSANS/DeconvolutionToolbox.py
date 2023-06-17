# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:17:32 2021

@author: bjh3
"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt



def RichLuc2D(p,d,Nits,reg):

    u = 0. + d
    
    for j in range(Nits):
        
        kr = np.real( d / (reg + fftConv2D(p,u) ))
        u = u * fftConv2D(kr,p)
    
    return u


def RichLucSelfN2D(d,N,Nits,reg):
    
    u = 0. + d
    #fp = np.fft.ifftn(p)
    #Nsq = (p.shape[0]*p.shape[1])
    for j in range(Nits):
        
        #kr = np.real( d / (1e-5*1.j + signal.convolve2d(p,u,mode = 'same') ))
        #u = u * signal.convolve2d(kr,p, mode = 'same')
        kr = np.real( d / (reg + fftConvN2D(u,N) ))
        u = u * fftConvKrN2D(kr,u,N-1)
        #kr = np.real(d /(1e-5*1.j + np.maximum(0,np.real( np.fft.fftn(np.fft.ifftn(u) * fp) ))))
        #u = u * np.maximum(0,np.abs( np.fft.fftn(np.fft.ifft(kr) * fp)))
    
    return u

def RichLucSelfNKer2D(p,d,N,Nits,reg, ld = False):
    
    u = RichLuc2D(p,d,Nits,reg)
    Nlog2 = int(np.log(N) / np.log(2)) - 1
    
    u = RichLucSelfN2D(u,N * 2**(-Nlog2),Nits,reg)
    for j in range(Nlog2):
        u = RichLucSelfN2D(u,2,Nits,reg)
   
    #u = 0. + d
    #fp = np.fft.ifftn(p)
    #Nsq = (p.shape[0]*p.shape[1])
    if ld:
        err = np.full(Nits,0.)
    for j in range(Nits):
        
        #kr = np.real( d / (1e-5*1.j + signal.convolve2d(p,u,mode = 'same') ))
        #u = u * signal.convolve2d(kr,p, mode = 'same')
        kr = np.real( d / (reg + fftConvNp2D(p,u,N)))
        u = u * fftConvKrNp2D(p,kr,u,N-1)
        if ld:
            err[j] = np.sum((fftConvKrN2D(kr,u,N) - p)**2)
        #kr = np.real(d /(1e-5*1.j + np.maximum(0,np.real( np.fft.fftn(np.fft.ifftn(u) * fp) ))))
        #u = u * np.maximum(0,np.abs( np.fft.fftn(np.fft.ifft(kr) * fp)))
    if ld:
        plt.figure()
        plt.plot(err)
    return u

def fftConvN2D(A,N):
    
    shp = np.array(A.shape) / 2
    
    #A = np.roll(A,(int(shp[0]),int(shp[1])),axis=(0,1))
    #B = np.roll(B,(int(shp[0]),int(shp[1])),axis=(0,1))
    
    return np.real(np.fft.ifftn(np.fft.fftn(A)**N))
    
    #return np.roll(C,(-int(shp[0]),-int(shp[1])),axis=(0,1))

def fftConvNp2D(p,A,N):
    
    shp = np.array(A.shape) / 2
    
    #A = np.roll(A,(int(shp[0]),int(shp[1])),axis=(0,1))
    #p = np.roll(p,(int(shp[0]),int(shp[1])),axis=(0,1))
    
    return np.real(np.fft.ifftn(np.fft.fftn(p)*np.fft.fftn(A)**N))
    
    #return np.roll(C,(-int(shp[0]),-int(shp[1])),axis=(0,1))


def fftConvKrNp2D(p,A,B,N):
    
    shp = np.array(A.shape) / 2
    
    #A = np.roll(A,(int(shp[0]),int(shp[1])),axis=(0,1))
    #B = np.roll(B,(int(shp[0]),int(shp[1])),axis=(0,1))
    #p = np.roll(p,(int(shp[0]),int(shp[1])),axis=(0,1))
    
    return np.real(np.fft.ifftn(np.fft.fftn(A) * np.fft.fftn(p) * np.fft.fftn(B)**N))
    
    #return np.roll(C,(-int(shp[0]),-int(shp[1])),axis=(0,1))

def fftConvKrN2D(A,B,N):
    
    shp = np.array(A.shape) / 2
    
    #A = np.roll(A,(int(shp[0]),int(shp[1])),axis=(0,1))
    #B = np.roll(B,(int(shp[0]),int(shp[1])),axis=(0,1))
    
    return np.real(np.fft.ifftn(np.fft.fftn(A) * np.fft.fftn(B)**N))
    
    #return np.roll(C,(-int(shp[0]),-int(shp[1])),axis=(0,1))

def fftConv2D(A,B):
    
    shp = np.array(A.shape) / 2
    
    #A = np.roll(A,(int(shp[0]),int(shp[1])),axis=(0,1))
    #B = np.roll(B,(int(shp[0]),int(shp[1])),axis=(0,1))
    
    return np.real(np.fft.fftn(np.fft.ifftn(A,norm = 'ortho') * np.fft.ifftn(B,norm = 'ortho')))
    
    #return np.roll(C,(-int(shp[0]),-int(shp[1])),axis=(0,1))

def DeconSignalSng(sIn,s0In,N, N2 = 1, Nits = 5, reg = 1e-6, ld = False,s=np.array([None])):
    
    if np.all(s) == None:
        s = 0. + sIn
        
    
    for j in range(s.shape[0]):
        if len(s0In.shape) == 2:
            s[j] = RichLucSelfNKer2D(s0In,s[j],N,Nits,reg, ld = ld)
        else:
            s[j] = RichLucSelfNKer2D(s0In[j],s[j],N,Nits,reg, ld = ld)
        
    
    if N2 > 1:
        
        for j in range(s.shape[0]):
            s[j] = RichLucSelfN2D(s[j],N2,Nits,reg)
    
    return s
        

def DeconSignal(sIn,s0In,fI00,N, Nits = 5, reg = 1e-6):
    
    I00 = np.fft.ifftn(fI00)
    
    s = np.real(np.fft.fftn(np.fft.ifftn(sIn,axes=(1,2)) / I00,axes = (1,2)))
    s0 = np.real(np.fft.fftn(np.fft.ifftn(s0In) / I00))
    
    s = s / np.sum(s,axis=(1,2))[:,None,None]
    s0 = s0 / np.sum(s0)
    
    for j in range(s.shape[0]):
        s[j] = RichLucSelfNKer2D(s0,s[j],N,Nits,reg)
        
    return s

def LwPs(A,sz,ordr):
    
    dims = np.shape(A)
     
    
    kx = np.roll(np.arange(-int(dims[1]/2),int(dims[1]/2)),-int(dims[1]/2))/dims[1]
    ky = np.roll(np.arange(-int(dims[0]/2),int(dims[0]/2)),-int(dims[0]/2))/dims[0]
    kfil = 1 / np.sqrt(1 + ((kx[None,:]**2 + ky[:,None]**2)*sz**2)**ordr)
    
    return np.real(np.fft.ifftn(np.fft.fftn(A)*kfil))



def InvMat(W,WT,Gss,Sng,lam,Nits):
    
    GssShape = Gss.shape
    SngShape = Sng.shape
    
    Gss = Gss.flatten()
    Sng = Sng.flatten()
    
    Nrx = csc_matrix((1/np.maximum(1, W.sum(axis = 0))).reshape(W.shape[1],1))
    Nry = csc_matrix((1/np.maximum(1, W.sum(axis = 1))).reshape(W.shape[0],1))
    
    Rcn = 0. + Gss
    
    #for j in range(Gss.shape[1]):
    SngSps = csc_matrix(Sng[:,None])
    GssSps = csc_matrix(Gss[:,None])

    for k in range(Nits):
        GssSps += lam * WT.dot((SngSps - W.dot(GssSps)).multiply(Nry)).multiply(Nrx)
        #if k % 100 ==0:
        #    Gss = np.clip(Gss,lw,hg)
    Rcn = GssSps.toarray()[:,0]
    
    Gss = Gss.reshape(GssShape)
    Rcn = Rcn.reshape(GssShape)
    Sng = Sng.reshape(SngShape)
    
    return Rcn