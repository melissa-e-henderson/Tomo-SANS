# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:54:37 2021

@author: bjh3
"""

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.offsetbox as obx

from . import SkyVis as SV

from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation


import colorsys

import sys

import time


# =============================================================================
# Import spin density reconstruction from real data
# =============================================================================

def ImportM(filename, typ = 1):
    
    #global Skx, Sky, Skz, Bx, By, Bz
    #if typ == 1
    hf=h5py.File(filename,'r')
    Skx=hf['m_recon'][1]
    Sky=hf['m_recon'][2]
    Skz=hf['m_recon'][0]
    #Bx=hf['Bx'][:,:,:]
    #By=hf['By'][:,:,:]
    #Bz=hf['Bz'][:,:,:]
    hf.close()
    
    return Skx, Sky, Skz


# =============================================================================
# Import spin density reconstruction from phantom
# =============================================================================

def ImportPhantM(filename):
    
    #global Skx, Sky, Skz
    
    hf=h5py.File(filename,'r')
    Skx=hf['phantom_parameters/Sk'][:,:,:,1]
    Sky=hf['phantom_parameters/Sk'][:,:,:,2]
    Skz=-hf['phantom_parameters/Sk'][:,:,:,0]
    hf.close()
    
    return Skx, Sky, Skz
    
# =============================================================================
# Progress Bar for when tqdm is not suitable    
# =============================================================================
    
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}]".format( "#"*block + "-"*(barLength-block)) + "{:.3f}".format(progress*100.) + "%"
    sys.stdout.write(text)
    sys.stdout.flush()


# =============================================================================
# Make RGB version of Skyrmion Lattice for Plotting
# =============================================================================
    
def MakeRGB(Skx, Sky, Skz):
    
    
    hls_to_rgb = np.vectorize(colorsys.hls_to_rgb) # retrieve colorspace
    
    dims = np.shape(Skx) # dimensions of lattice
    
    phi = np.arctan2(Sky,Skx) # polar angle
    
    RGBA = np.full((dims[0],dims[1],dims[2],3),0.) #initiate output
    RGBA = hls_to_rgb((phi)/(2*np.pi)+0.5,np.clip((-Skz+1)/2,0,1),1) # Compute color based on phi and Skz
    RGBA = np.transpose(RGBA,(1,2,3,0)) # Transpose for formatting
    
    return RGBA


# =============================================================================
# Find Free Energy defects and note them in RGB format
# =============================================================================

def HToRGB(Htot):
    
    global Hrgb
    
    dfct = 3.5 # Defect Threshold
    
    dims = np.shape(Htot) #shape of reconstruction
    Hrgb = np.full((dims[0],dims[1],dims[2],4),0.) #Declare RGB version with alpha channel set to clear
    pnk = np.array([0,0,0,1]) # opaque
    Hrgb = Hrgb + np.sign(Htot[:,:,:,None]-np.clip(Htot[:,:,:,None],None,dfct))*pnk[None,None,None,:] # populate deflects
    
    return Hrgb

# =============================================================================
# Make and Save Movie of Reconstruction or Phantom
# =============================================================================
    
def AnimateS(savename, Skx, Sky, Skz, frms = 0, comp = True, Kappa = -0.39, spct = 1):
    
    if frms == 0:
        frms = Skx.shape[0]
    
    ### Inputs ###
    ##############
    # savename - file name to save
    # frms - frames in movie - should be same as depth of reconstruction
    # comp - 0 if emergent fields and free energy need to be computed
    
    ### Global Inputs ###
    #####################
    # Skx, Sky, Skz - spin density
    # rhoS - z-component of emergent field / topological charge density
    # Hrgb - free energy defects
    # Inputs of ChDen(), CompH(), MakeRGB(), and HToRGB()
    
    #global rhoS, Hrgb
    
    ### Global Outputs ###
    ######################
    # ani - animation object
    # Outputs of ChDen(), CompH(), MakeRGB(), and HToRGB()
    
    global ani
    
    if comp:
        print('Computing Emergent Fields and Free Energy....')
        #ChDen() # Topological Charge Density
        #CompH() # Free Energy
        rhoM, bz, bx, by, rhoEm, HHeis, HDM = SV.CalcMetaParams(Skx, Sky, Skz)
        RGBA = MakeRGB(Skx, Sky, Skz) # Computes colors for plotting skyrmion lattice is HLS colorspace
        Hrgb = HToRGB(HHeis + 4 * Kappa * HDM ) # Makes an RGB version of the Free Energy
    else:
        MakeRGB() # Computes colors for plotting skyrmion lattice is HLS colorspace
        HToRGB() # Makes an RGB version of the Free Energy
    
    g1 = gaussian_filter(bz,sigma=(0,1,1)) # Filter rhoS before plotting
    
    dims = np.shape(Skx) #dimensions
    
    X = np.linspace(0,(dims[1]-1)*0.0143,dims[1]) # x coordinates
    
    dp = 0 # depth index
    
    fig, (ax2,ax) = plt.subplots(1, 2, figsize=(16.5,7.5)) # Make Figure
    
    #### Setup Axes ###
    ###################
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    ax.grid(True)
    ax2.grid(True)
    fig.tight_layout(pad = 5.05)
    
    ax.set_xlabel('Position, [$\mu$m]',fontsize=20)
    ax2.set_ylabel('Position, [$\mu$m]',fontsize=20)
    ax.tick_params(labelsize=20)
    
    ax2.set_xlabel('Position, [$\mu$m]',fontsize=20)
    ax2.tick_params(labelsize=20)
    
    fig.subplots_adjust(right=0.88)
    
    # Extent of axes
    extn = (-0.5*0.0143,(dims[1]-0.5)*0.0143,-0.5*0.0143,(dims[1]-0.5)*0.0143)
    
    # Plot the topological charge density
    sz = ax.imshow(g1[dp,:,:],cmap='Spectral_r',origin='lower',extent = extn, vmin = -0.01, vmax = 0.01, animated=True)  
    
    # Make colorbar #
    cbar_ax = fig.add_axes([0.9,0.11,0.02,0.45])
    cbar = fig.colorbar(sz, cax = cbar_ax, pad = 0.02,aspect = 5)
    cbar.ax.set_title('    $\\rho_S$',fontsize=20, pad = 18)
    cbar.ax.tick_params(labelsize=20)
    
    # Add Image of key showing magnetic field direction #
    ky_ax = fig.add_axes([0.93,0.79,0.025,0.025])
    MagKey = plt.imread('MagKeyDef.png', format = 'png')
    ob = obx.OffsetImage(MagKey, zoom = 0.3)
    ab = obx.AnnotationBbox(ob, (0,0), frameon=False)
    ky_ax.add_artist(ab)
    
    ### Setup Animation Parameters ###
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=24000)
    
    print('Video Progress:')
    
    def UpdateImage(dp):  # Updates Image as a function of depth
        
        #Clear Axes
        ax.clear()
        ax2.clear()
        
        # Setup Axes Labels
        ax.set_xlabel('Position, [$\mu$m]',fontsize=20)
        ax2.set_ylabel('Position, [$\mu$m]',fontsize=20)
        ax2.set_xlabel('Position, [$\mu$m]',fontsize=20)
        
        depth = np.round(dp * 0.0143 * spct, decimals = 3) #Current depth
        titlestring = 'Depth: ' + '{:.3f}'.format(depth) + ' $\mu$m'
        fig.suptitle(titlestring, fontsize=20) # Print Depth in title
        
        # Plot Topological Charge Density / bz
        ax.imshow(g1[dp,:,:],cmap='Spectral_r',origin='lower',extent = (-0.5*0.0143,127.5*0.0143,-0.5*0.0143,127.5*0.0143), vmin = -0.05, vmax = 0.05,animated=True)
        # Superimpose Energetic Defects
        ax.imshow( Hrgb[dp,:,:,:], origin='lower',extent = extn,animated=True)
        # Add Vectors
        ax.quiver(X,X,Sky[dp,:,:], Skx[dp,:,:])
        
        # Skyrmion Lattice
        ax2.imshow(RGBA[dp,:,:,:],origin='lower',extent = extn,animated=True)
        
        #ax2.quiver(X,X,Sky[dp,:,:] * 5*10**3, Skx[dp,:,:] * 5*10**3)
        update_progress(dp/(frms+0.))
        
    
    # Perform Animation
    ani = animation.FuncAnimation(fig, UpdateImage, frames = frms, interval = 100, blit = False, repeat = False, save_count=10)
    
    # Save Animation
    ani.save(savename, writer = writer)
    
    plt.close()
    
    
def AnimateSC(savename, m, frms = 0, comp = True, Kappa = -0.39, spct = 1, RGBSgn = -1, fps = 10):
    
    if frms == 0:
        frms = m[0].shape[0] 
    
    ### Inputs ###
    ##############
    # savename - file name to save
    # frms - frames in movie - should be same as depth of reconstruction
    # comp - 0 if emergent fields and free energy need to be computed
    
    ### Global Inputs ###
    #####################
    # Skx, Sky, Skz - spin density
    # rhoS - z-component of emergent field / topological charge density
    # Hrgb - free energy defects
    # Inputs of ChDen(), CompH(), MakeRGB(), and HToRGB()
    
    #global rhoS, Hrgb
    
    ### Global Outputs ###
    ######################
    # ani - animation object
    # Outputs of ChDen(), CompH(), MakeRGB(), and HToRGB()
    
    global ani
    
    if comp:
        print('Computing Emergent Fields and Free Energy....')
        #ChDen() # Topological Charge Density
        #CompH() # Free Energy
        rhoM, bz, bx, by, rhoEm, HHeis, HDM = SV.CalcMetaParams(m)
        RGBA = MakeRGB(m[1], m[2], RGBSgn * m[0]) # Computes colors for plotting skyrmion lattice is HLS colorspace
        #Hrgb = HToRGB(HHeis + 4 * Kappa * HDM ) # Makes an RGB version of the Free Energy
    else:
        MakeRGB() # Computes colors for plotting skyrmion lattice is HLS colorspace
        HToRGB() # Makes an RGB version of the Free Energy
    
    dims = np.shape(m[0]) #dimensions
    
    X = np.linspace(0,(dims[1]-1)*0.0143,dims[1]) # x coordinates
    
    dp = 0 # depth index
    
    fig, (ax2,ax) = plt.subplots(1, 2, figsize=(16.5,7.5)) # Make Figure
        
    #### Setup Axes ###
    ###################
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    ax.grid(True)
    ax2.grid(True)
    fig.tight_layout(pad = 5.05)
    
    ax.set_xlabel('Position, [$\mu$m]',fontsize=20)
    ax2.set_ylabel('Position, [$\mu$m]',fontsize=20)
    
    
    ax2.set_xlabel('Position, [$\mu$m]',fontsize=20)
    ax.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    
    fig.subplots_adjust(right=0.88)
    
    # Extent of axes
    extn = (-0.5*0.0143,(dims[1]-0.5)*0.0143,-0.5*0.0143,(dims[1]-0.5)*0.0143)
    
    lvls = np.linspace(-1,1,5)
    
    # Plot the emergent charge density
    sz = ax.imshow(rhoEm[dp],cmap='seismic',origin='lower',extent = extn, vmin = -0.05, vmax = 0.05, animated=True)  
    cn = ax.contour(m[0,dp],lvls, extent = extn, origin = 'lower', colors = 'black')
    # Make colorbar #
    cbar_ax = fig.add_axes([0.9,0.11,0.02,0.45])
    cbar = fig.colorbar(sz, cax = cbar_ax, pad = 0.02,aspect = 5)
    cbar.ax.set_title('    $\\rho_{E}$',fontsize=20, pad = 18)
    cbar.ax.tick_params(labelsize=20)
    
    sz2 = ax2.imshow(RGBA[dp,:,:,:],origin='lower',extent = extn,animated=True)
    
    # Add Image of key showing magnetic field direction #
    ky_ax = fig.add_axes([0.93,0.79,0.025,0.025])
    MagKey = plt.imread('MagKeyDef.png', format = 'png')
    ob = obx.OffsetImage(MagKey, zoom = 0.3)
    ab = obx.AnnotationBbox(ob, (0,0), frameon=False)
    ky_ax.add_artist(ab)
    
    
    ### Setup Animation Parameters ###
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps = 10, metadata=dict(artist='Me'), bitrate=-1)
    writer = animation.FFMpegWriter(fps = fps)
    tm = time.time()
    print('Video Progress:')
    #update_progress(0)
    
    def UpdateImage(dp):  # Updates Image as a function of depth
        
        
    
        #Clear Axes
        ax.clear()
        ax2.clear()
        
        # Setup Axes Labels
        ax.set_xlabel('Position, [$\mu$m]',fontsize=20)
        ax2.set_ylabel('Position, [$\mu$m]',fontsize=20)
        ax2.set_xlabel('Position, [$\mu$m]',fontsize=20)
        ax.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        
        depth = np.round(dp * 0.0143 * spct, decimals = 3) #Current depth
        titlestring = 'Depth: ' + '{:.3f}'.format(depth) + ' $\mu$m'
        fig.suptitle(titlestring, fontsize=20) # Print Depth in title
        
        # Plot Topological Charge Density / bz
        sz = ax.imshow(rhoEm[dp],cmap='seismic',origin='lower',extent = extn, vmin = -0.05, vmax = 0.05,animated=True)
        # Superimpose Energetic Defects
        # ax.imshow( Hrgb[dp,:,:,:], origin='lower',extent = extn,animated=True)
        # Add Vectors
        cn = ax.contour(m[0,dp],lvls, extent = extn, origin = 'lower', colors = 'black')
        
        # Skyrmion Lattice
        sz2 = ax2.imshow(RGBA[dp,:,:,:],origin='lower',extent = extn,animated=True)
        
        
        #ax2.quiver(X,X,Sky[dp,:,:] * 5*10**3, Skx[dp,:,:] * 5*10**3)
        update_progress(dp/(frms+0.))
        
        return sz2
        
    
    # Perform Animation
    ani = animation.FuncAnimation(fig, UpdateImage, frames = frms, blit = False, repeat = False, cache_frame_data = False)
    
    # Save Animation
    ani.save(savename, writer = writer)
    
    plt.close()
    tf = time.time()
    print(tf - tm)
    
def AnimateSCRGB(savename, S, frms = 0, RGBSgn = -1, fps = 10):
    
    if frms == 0:
        frms = S.m[0].shape[0] 
    
    ### Inputs ###
    ##############
    # savename - file name to save
    # frms - frames in movie - should be same as depth of reconstruction
    # comp - 0 if emergent fields and free energy need to be computed
    
    ### Global Inputs ###
    #####################
    # Skx, Sky, Skz - spin density
    # rhoS - z-component of emergent field / topological charge density
    # Hrgb - free energy defects
    # Inputs of ChDen(), CompH(), MakeRGB(), and HToRGB()
    
    #global rhoS, Hrgb
    
    ### Global Outputs ###
    ######################
    # ani - animation object
    # Outputs of ChDen(), CompH(), MakeRGB(), and HToRGB()
    
    global ani
    
    RGBB = SV.MakeRGB(np.abs(S.rhoEm),S.rhoEm/np.max(S.rhoEm),S.dm[0,0]/np.max(S.dm[0,0]))
    RGBA = MakeRGB(S.m[1], S.m[2], RGBSgn * S.m[0])
    
    dims = np.shape(S.m[0]) #dimensions
    
    X = np.linspace(0,(dims[1]-1)*0.0143,dims[1]) # x coordinates
    
    dp = 0 # depth index
    
    fig, (ax2,ax) = plt.subplots(1, 2, figsize=(16.5,7.5)) # Make Figure
        
    #### Setup Axes ###
    ###################
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    ax.grid(True)
    ax2.grid(True)
    fig.tight_layout(pad = 5.05)
    
    ax.set_xlabel('Position, [$\mu$m]',fontsize=20)
    ax2.set_ylabel('Position, [$\mu$m]',fontsize=20)
    
    
    ax2.set_xlabel('Position, [$\mu$m]',fontsize=20)
    ax.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    
    fig.subplots_adjust(right=0.88)
    
    # Extent of axes
    extn = (-0.5*0.0143,(dims[1]-0.5)*0.0143,-0.5*0.0143,(dims[1]-0.5)*0.0143)
    
    lvls = [0,] #np.linspace(-1,1,3)
    
    # Plot the emergent charge density
    #sz = ax.imshow(rhoEm[dp],cmap='seismic',origin='lower',extent = extn, vmin = -0.05, vmax = 0.05, animated=True)  
    
    sz = ax.imshow(RGBB[dp], extent = extn, origin = 'lower')
    cn = ax.contour(S.m[0,dp],lvls, extent = extn, origin = 'lower', colors = 'white')
    # Make colorbar #
    #cbar_ax = fig.add_axes([0.9,0.11,0.02,0.45])
    #cbar = fig.colorbar(sz, cax = cbar_ax, pad = 0.02,aspect = 5)
    #cbar.ax.set_title('    $\\rho_{E}$',fontsize=20, pad = 18)
    #cbar.ax.tick_params(labelsize=20)
    
    sz2 = ax2.imshow(RGBA[dp,:,:,:],origin='lower',extent = extn,animated=True)
    
    # Add Image of key showing magnetic field direction #
    ky_ax = fig.add_axes([0.93,0.49,0.025,0.05])
    MagKey = plt.imread('MagEmKey.png', format = 'png')
    ob = obx.OffsetImage(MagKey, zoom = 0.3)
    ab = obx.AnnotationBbox(ob, (0,0), frameon=False)
    ky_ax.add_artist(ab)
    ky_ax.axis('off')
    
    
    ### Setup Animation Parameters ###
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps = 10, metadata=dict(artist='Me'), bitrate=-1)
    writer = animation.FFMpegWriter(fps = fps)
    tm = time.time()
    print('Video Progress:')
    #update_progress(0)
    
    def UpdateImage(dp):  # Updates Image as a function of depth
        
        
    
        #Clear Axes
        ax.clear()
        ax2.clear()
        
        # Setup Axes Labels
        ax.set_xlabel('Position, [$\mu$m]',fontsize=20)
        ax2.set_ylabel('Position, [$\mu$m]',fontsize=20)
        ax2.set_xlabel('Position, [$\mu$m]',fontsize=20)
        ax.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        
        depth = np.round(dp * 0.0143 * S.voxelaspect, decimals = 3) #Current depth
        titlestring = 'Depth: ' + '{:.3f}'.format(depth) + ' $\mu$m'
        fig.suptitle(titlestring, fontsize=20) # Print Depth in title
        
        # Plot Topological Charge Density / bz
        #sz = ax.imshow(rhoEm[dp],cmap='seismic',origin='lower',extent = extn, vmin = -0.05, vmax = 0.05,animated=True)
        sz = ax.imshow(RGBB[dp],extent = extn, origin = 'lower')
        # Superimpose Energetic Defects
        # ax.imshow( Hrgb[dp,:,:,:], origin='lower',extent = extn,animated=True)
        # Add Vectors
        cn = ax.contour(S.m[0,dp],lvls, extent = extn, origin = 'lower', colors = 'white')
        
        # Skyrmion Lattice
        sz2 = ax2.imshow(RGBA[dp,:,:,:],origin='lower',extent = extn,animated=True)
        
        
        #ax2.quiver(X,X,Sky[dp,:,:] * 5*10**3, Skx[dp,:,:] * 5*10**3)
        update_progress(dp/(frms+0.))
        
        return sz2
        
    
    # Perform Animation
    ani = animation.FuncAnimation(fig, UpdateImage, frames = frms, blit = False, repeat = False, cache_frame_data = False)
    
    # Save Animation
    ani.save(savename, writer = writer)
    
    plt.close()
    tf = time.time()
    print(tf - tm)