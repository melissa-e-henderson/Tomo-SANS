# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:08:53 2021

@author: bjh3
"""


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import colorsys

from scipy.ndimage import gaussian_filter

def SlideDepth(A, cmap = 'jet', mx = None, mn = None, aspect = 1):
    
    dims = np.array(A.shape) # Dimensins of input
    cent = (dims/2).astype(int) # Center of input
    fig, ax = plt.subplots() # initiate figure
    
    #range of plot
    if mx == None:
        mx = np.max(A)
    if mn == None:
        mn = np.min(A)
    #mx = np.max(A)
    
    #initiate plot
    img = ax.imshow(A[cent[0],:,:],vmin = mn, vmax = mx, cmap = cmap, origin = 'lower', aspect = aspect)
    
    # Add ColorBar
    cbar_ax = fig.add_axes([0.9,0.2,0.02,0.5])
    cbar = fig.colorbar(img, cax = cbar_ax, pad = 0.02,aspect = 5)
    
    #Initiate Slider
    axdpth = plt.axes([0.15,0.01,0.65,0.03])
    sdpth = Slider(axdpth,'Depth', 0, dims[0]-1, valinit = cent[0], valstep = 1)
    
    #updateplot
    def update(val):
        
        dp = int(sdpth.val) # Depth from slider
        ax.imshow(A[dp,:,:],vmin = mn, vmax = mx, cmap = cmap, origin='lower', aspect = aspect) # Print new plot
        fig.canvas.draw_idle()
        #plt.pause(0.1) # Makes run smoother
    
    
    sdpth.on_changed(update) #update plot on slider change
    
    plt.show()
    
    return sdpth

def SlideDepthCont(A, B, arr=2, cmp = 'seismic', mx = None, mn = None, mxC = None, mnC = None, NC = 5, clrs = 'black', Symm = 'True'):
    
    szx = A.shape[2]
    dims = np.array(A.shape) # Dimensins of input
    cent = (dims/2).astype(int) # Center of input
    fig, ax = plt.subplots() # initiate figure
    
    #range of plot
    if mx == None:
        mx = np.max(A)
    if mn == None:
        mn = np.min(A)
    
    if Symm:
        mx = np.maximum(mx,-mn)
        mn = - mx
    
    if mxC == None:
        mxC = np.max(B)
    if mnC == None:
        mnC = np.min(B)
    
    Cntrs = np.linspace(mnC,mxC,NC)
    
    #initiate plot
    img = ax.imshow(A[cent[0],:,:],vmin = mn, vmax = mx, origin = 'lower', cmap = cmp)
    
    
    # Add ColorBar
    cbar_ax = fig.add_axes([0.9,0.2,0.02,0.5])
    cbar = fig.colorbar(img, cax = cbar_ax, pad = 0.02,aspect = 5)
    
    Cont = ax.contour(B[cent[0]],Cntrs,colors = clrs)
    
    #Initiate Slider
    axdpth = plt.axes([0.15,0.01,0.65,0.03])
    sdpth = Slider(axdpth,'Depth', 0, dims[0]-1, valinit = cent[0], valstep = 1)
    
    #updateplot
    def update(val):
        
        dp = int(sdpth.val) # Depth from slider
        ax.clear()
        ax.imshow(A[dp,:,:],vmin = mn, vmax = mx, origin = 'lower', cmap = cmp) # Print new plot
        ax.contour(B[dp],Cntrs,colors = clrs)
        fig.canvas.draw_idle()
        #plt.pause(0.5) # Makes run smoother
        
        
        
        
    
    
    sdpth.on_changed(update) #update plot on slider change
    
    plt.show()
    
    return sdpth



def SlideDepthVec(mvec, arr=2, cmp = 'jet', mx = None, mn = None):
    
    A = mvec[0]
    szx = A.shape[2]
    dims = np.array(A.shape) # Dimensins of input
    cent = (dims/2).astype(int) # Center of input
    fig, ax = plt.subplots() # initiate figure
    
    #range of plot
    if mx == None:
        mx = np.max(A)
    if mn == None:
        mn = np.min(A)
    
    #initiate plot
    img = ax.imshow(A[cent[0],:,:],vmin = mn, vmax = mx, origin = 'lower', cmap = cmp)
    
    
    # Add ColorBar
    cbar_ax = fig.add_axes([0.9,0.2,0.02,0.5])
    cbar = fig.colorbar(img, cax = cbar_ax, pad = 0.02,aspect = 5)
    
    qvr = ax.quiver(np.arange(0,szx,arr)[:,None],np.arange(0,szx,arr)[None,:],mvec[2,cent[0],::arr,::arr],mvec[1,cent[0],::arr,::arr])
    
    #Initiate Slider
    axdpth = plt.axes([0.15,0.01,0.65,0.03])
    sdpth = Slider(axdpth,'Depth', 0, dims[0]-1, valinit = cent[0], valstep = 1)
    
    #updateplot
    def update(val):
        
        dp = int(sdpth.val) # Depth from slider
        ax.clear()
        ax.imshow(A[dp,:,:],vmin = mn, vmax = mx, origin = 'lower', cmap = cmp) # Print new plot
        ax.quiver(np.arange(0,szx,arr)[:,None],np.arange(0,szx,arr)[None,:],mvec[2,dp,::arr,::arr],mvec[1,dp,::arr,::arr])
        fig.canvas.draw_idle()
        #plt.pause(0.5) # Makes run smoother
        
        
        
        
    
    
    sdpth.on_changed(update) #update plot on slider change
    
    plt.show()
    
    return sdpth

def MakeScatt(A,B,scale = 'log',Alab = None, Blab = None, xyrange = None, vmax = None,nbns = 200, wts = np.array(None), density = True, rescale = 1, rtn = None):
    x = A.flatten()
    y = B.flatten()
    
    if wts.all() == None:
        wts = None
    else:
        wts = wts.flatten()
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=nbns, density = density, range = xyrange, weights = wts)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig = plt.figure()
    plt.clf()
    if scale == 'lin':
        plt.imshow(heatmap.T*rescale, extent = extent, origin='lower', aspect = 'auto', cmap= 'jet', vmin = 0, vmax = vmax)
    else:
        plt.imshow(np.log10(1+heatmap.T*rescale), extent = extent, origin='lower', aspect = 'auto', cmap = 'jet', vmin = 0, vmax = vmax)
    plt.xlabel(Alab)
    plt.ylabel(Blab)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    if rtn == 'fig':
        return fig
    elif rtn == 'hist':
        return heatmap, xedges, yedges

def Grad2(m,Lm = np.array(None), spct = 1):
    if Lm.shape ==():
        Lm = np.full((3,3,m.shape[1],m.shape[2],m.shape[3]),0.)
    else:
        Lm *= 0.
    for i in range(2):
        Lm[:,0,:] += np.roll(m,2*i-1,axis=1) / spct**2
    Lm[:,0,:] += - 2/spct**2 * m
    for j in range(2):
        for i in range(2):
            Lm[:,j+1,:] += np.roll(m,2*i-1,axis=j+2)
        Lm[:,j+1,:] += - 2 * m
    
    return Lm

def Lap(m,Lm = np.array(None), spct = 1):
    if Lm.shape ==():
        Lm = np.full(m.shape,0.)
    else:
        Lm *= 0.
    for i in range(2):
        Lm += np.roll(m,2*i-1,axis=1) / spct**2
    for j in range(2):
        for i in range(2):
            Lm += np.roll(m,2*i-1,axis=j+2)
    Lm += - (4 + 2/spct**2) * m
    return Lm

def Grad(m,dm = np.array(None), spct = 1):
    if dm.shape == ():
        dm = np.full((3,3,m.shape[1],m.shape[2],m.shape[3]),0.)
    for j in range(3):
        dm[:,j,:] = 0.5 * (np.roll(m,-1,axis = j+1) - np.roll(m,1,axis = j+1))
    dm[:,0] *= 1/spct
    return dm
    
def Curl(m,spct = 1):
    dm = Grad(m,spct = spct)
    
    return np.stack((dm[2,1]-dm[1,2],dm[0,2]-dm[2,0],dm[1,0]-dm[0,1]))

def CalcMetaParams(m, Method = 'FiniteDiff', spct = 1):
    
    ### Inputs ###
    ##############
    # Skx, Sky, Skz - x, y, and z components of spin density
    # Kappa = D / (2 J), skyrmion wave vector in units of inverse pixels
    
    ### Global Outputs ###
    ######################
    # rhoM - magnetic charge density
    # rhoS - z component of emergent charge density (also topological charge density)
    # bx, by, bz - x, y, and z components of emergent field
    # rhoEm - erergent charge density
    # mij - second derivatives of spin density
    # HHeis - Heisenberg exchange Term in free energy density
    # HDM - DM term in Free energy density
    
    if Method == 'Fourier':
        dims = m[0].shape # dimensions of spin density
        
        ### Make Derivative Operators ###
        #################################
        qz = np.linspace(0,dims[0]-1,dims[0]) - dims[0]/2
        qx = np.linspace(0,dims[1]-1,dims[1]) - dims[1]/2
        qy = np.linspace(0,dims[2]-1,dims[2]) - dims[2]/2
        
        qz = np.roll(qz, -int(dims[0]/2))
        qx = np.roll(qx, -int(dims[1]/2))
        qy = np.roll(qy, -int(dims[2]/2))
        
        
        # Fourier transform of spin density
        fmx = np.fft.fftn(m[1])
        fmy = np.fft.fftn(m[2])
        fmz = np.fft.fftn(m[0])
        
        # Compute second order derivatives #
        ####################################
        mxz = -np.imag(np.fft.ifftn(fmx * qz[:,None,None]))* (2*np.pi / dims[0])
        mxx = -np.imag(np.fft.ifftn(fmx * qx[None,:,None]))* (2*np.pi / dims[1])
        mxy = -np.imag(np.fft.ifftn(fmx * qy[None,None,:]))* (2*np.pi / dims[2])
        
        myz = -np.imag(np.fft.ifftn(fmy * qz[:,None,None]))* (2*np.pi / dims[0])
        myx = -np.imag(np.fft.ifftn(fmy * qx[None,:,None]))* (2*np.pi / dims[1])
        myy = -np.imag(np.fft.ifftn(fmy * qy[None,None,:]))* (2*np.pi / dims[2])
    
        mzz = -np.imag(np.fft.ifftn(fmz * qz[:,None,None]))* (2*np.pi / dims[0])
        mzx = -np.imag(np.fft.ifftn(fmz * qx[None,:,None]))* (2*np.pi / dims[1])
        mzy = -np.imag(np.fft.ifftn(fmz * qy[None,None,:]))* (2*np.pi / dims[2])
    
    elif Method == 'FiniteDiff':
        
        dm = Grad(m, spct = spct)
        
        mxx = dm[1,1]
        mxy = dm[1,2]
        mxz = dm[1,0]
        
        mzx = dm[0,1]
        mzy = dm[0,2]
        mzz = dm[0,0]
        
        myx = dm[2,1]
        myy = dm[2,2]
        myz = dm[2,0]
        
        Lm = Lap(m)
        
    else:
        #dmx = np.gradient()
        
        mxz = gaussian_filter(m[1], (0.5,0,0), order = (1,0,0), mode = 'wrap')
        mxx = gaussian_filter(m[1], (0,0.5,0), order = (0,1,0), mode = 'wrap')
        mxy = gaussian_filter(m[1], (0,0,0.5), order = (0,0,1), mode = 'wrap')
        
        myz = gaussian_filter(m[2], (0.5,0,0), order = (1,0,0), mode = 'wrap')
        myx = gaussian_filter(m[2], (0,0.5,0), order = (0,1,0), mode = 'wrap')
        myy = gaussian_filter(m[2], (0,0,0.5), order = (0,0,1), mode = 'wrap')
        
        mzz = gaussian_filter(m[0], (0.5,0,0), order = (1,0,0), mode = 'wrap')
        mzx = gaussian_filter(m[0], (0,0.5,0), order = (0,1,0), mode = 'wrap')
        mzy = gaussian_filter(m[0], (0,0,0.5), order = (0,0,1), mode = 'wrap')
    # Magnetic charge density #
    ###########################
    rhoM = mxx + myy + mzz  
    
    ### Emergent Field ###
    ######################
    
    #z component emergent field / topological charge density
    bz = ( m[1] * (myx * mzy - myy * mzx) + m[2] * (mzx * mxy - mxx * mzy) + 
          m[0] * ( mxx * myy - mxy*myx) ) / (4 * np.pi)
    
    #x componenent emergent field
    bx = ( m[1] * (myy * mzz - myz * mzy) + m[2] * (mzy * mxz - mxy * mzz) + 
          m[0] * ( mxy * myz - mxz*myy) ) / (4 * np.pi)
    
    #y component emergent field
    by = ( m[1] * (myz * mzx - myx * mzz) + m[2] * (mzz * mxx - mxz * mzx) + 
          m[0] * ( mxz * myx - mxx*myz) ) / (4 * np.pi)
    
    
    ### Emergent Charge Density ###
    ###############################
    if Method == 'Fourier':
        fbz = np.fft.fftn(bz)
        fby = np.fft.fftn(by)
        fbx = np.fft.fftn(bx)
        
        rhoEm = -np.imag(np.fft.ifftn(fbz * qz[:,None,None] * (2*np.pi / dims[0]) + fbx * qx[None,:,None] * (2*np.pi / dims[1]) + fby * qy[None,None,:] * (2*np.pi / dims[2])))
    else:
        rhoEm = 0*bz
        #for j in range(3):
        rhoEm += 0.5 * (np.roll(bz,-1,axis = 0) - np.roll(bz,1,axis = 0)) / spct
        rhoEm += 0.5 * (np.roll(bx,-1,axis = 1) - np.roll(bx,1,axis = 1))
        rhoEm += 0.5 * (np.roll(by,-1,axis = 2) - np.roll(by,1,axis = 2))
        #rhoEm = gaussian_filter(bz, (0.5,0,0), order = (1,0,0), mode = 'wrap') + gaussian_filter(bx, (0,0.5,0), order = (0,1,0), mode = 'wrap') + gaussian_filter(by, (0,0,0.5), order = (0,0,1), mode = 'wrap')
    '''
    if Method == 'FiniteDiff':
        HHeis = - np.sum(m * Lm,axis=0)
    else:
        # Heisenburg Exchange Term
    '''
    HHeis = mxx**2 + myy**2 + mzz**2 + mxy**2 + mxz**2 + myx**2 + myz**2 + mzx**2 + mzy**2
    
    # DM Term
    HDM = m[0]*(myx-mxy) + m[1]*(mzy-myz) + m[2]*(mxz-mzx)
    

    return rhoM, bz, bx, by, rhoEm, HHeis, HDM


def MakeRGB(A,Vx,Vy,mx = None,mn=None,shft = 0):
    
    if mx == None:
        mx = np.max(A)
    if mn ==None:
        mn = np.min(A)
    
    hls_to_rgb = np.vectorize(colorsys.hls_to_rgb) # retrieve colorspace
    
    #dims = np.shape(Skx) # dimensions of lattice
    
    phi = np.arctan2(Vy,Vx) # polar angle
    phi = np.mod(phi+shft,2*np.pi)
    
    RGBA = np.full(A.shape + (3,),0.) #initiate output
    RGBA = hls_to_rgb((phi)/(2*np.pi),np.clip((A-mn)/(mx-mn),0,1),1) # Compute color based on phi and Skz
    RGBA = np.transpose(RGBA,(1,2,3,0)) # Transpose for formatting
    
    return RGBA

def SlideDepthSpinDensMetas(S, CompMets = False):
    
    if CompMets:
        
        S.ComputeMetas()
    
    return SlideDepthContRGB(np.abs(S.rhoEm),S.rhoEm/np.max(S.rhoEm),S.dm[0,0]/np.max(S.dm[0,0]),S.m[0])

def SlideDepthContRGB(A, Vx, Vy, B, arr=2, mx = None, mn = None, mxC = None, mnC = None, NC = 3, clrs = 'white', linestyles = 'solid', Symm = False,shft = 0):
    
    #szx = A.shape[2]
    dims = np.array(A.shape) # Dimensins of input
    cent = (dims/2).astype(int) # Center of input
    fig, ax = plt.subplots() # initiate figure
    
    #range of plot
    if mx == None:
        mx = np.max(A)
    if mn == None:
        mn = np.min(A)
    
    if Symm:
        mx = np.maximum(mx,-mn)
        mn = - mx
    
    RGBA = MakeRGB(A,Vx,Vy,mx = mx, mn = mn,shft = 0)
    
    if mxC == None:
        mxC = np.max(B)
    if mnC == None:
        mnC = np.min(B)
    
    Cntrs = np.linspace(mnC,mxC,NC)
    
    #initiate plot
    img = ax.imshow(RGBA[cent[0]],vmin = mn, vmax = mx, origin = 'lower')
    
    
    # Add ColorBar
    #cbar_ax = fig.add_axes([0.9,0.2,0.02,0.5])
    #cbar = fig.colorbar(img, cax = cbar_ax, pad = 0.02,aspect = 5)
    
    Cont = ax.contour(B[cent[0]],Cntrs,colors = clrs, linestyles = linestyles)
    
    #Initiate Slider
    axdpth = plt.axes([0.15,0.01,0.65,0.03])
    sdpth = Slider(axdpth,'Depth', 0, dims[0]-1, valinit = cent[0], valstep = 1)
    
    #updateplot
    def update(val):
        
        dp = int(sdpth.val) # Depth from slider
        ax.clear()
        ax.imshow(RGBA[dp],vmin = mn, vmax = mx, origin = 'lower') # Print new plot
        ax.contour(B[dp],Cntrs,colors = clrs, linestyles = linestyles)
        fig.canvas.draw_idle()
        #plt.pause(0.5) # Makes run smoother
        
        
        
        
    
    
    sdpth.on_changed(update) #update plot on slider change
    
    plt.show()
    
    return sdpth