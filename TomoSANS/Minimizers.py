# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 08:42:02 2021

@author: bjh3
"""


import numpy as np
import h5py
from .DataPreprocess import *
from tqdm import tqdm



def SaveProgress(flnm, S, Cost, X, lm, init = False, Chi = True):
    
    if init:
        hf = h5py.File(flnm, 'a')
        mhf = hf.create_dataset('m',shape = S.m.shape, dtype = np.float32, data = S.m)
        Costhf = hf.create_dataset('Cost',shape = Cost.shape, dtype = np.float32, data = Cost)
        if Chi:
            Ikthf = hf.create_dataset('Ikt',shape = X.Ikt.shape, dtype = np.float32, data = X.Ikt)
            fImhf = hf.create_dataset('fIm',shape = X.fIm.shape, dtype = np.float32, data = X.fIm)
            fI0hf = hf.create_dataset('fI0',shape = X.fI0.shape, dtype = np.float32, data = X.fI0)
            wtshf = hf.create_dataset('wts',shape = X.weights.shape, dtype = np.float32, data = X.weights)
            ProjThhf = hf.create_dataset('ProjTh',shape = X.ProjTh.shape, dtype = np.float32, data = X.ProjTh)
            ProjPsihf = hf.create_dataset('ProjPsi',shape = X.ProjPsi.shape, dtype = np.float32, data = X.ProjPsi)
            hf.create_dataset('NzSample', data = X.N)
        #lmhf = hf.create_dataset('lm', data = lm)
        hf.close()
    
    else:
    
        hf = h5py.File(flnm, 'r+')
        mhf = hf['m']
        Costhf = hf['Cost']
        if Chi:
            Ikthf = hf['Ikt']
            Ikthf[:] = X.Ikt
        mhf[:] = S.m
        Costhf[:] = Cost
        
        hf.close()

    

def LLG(S,OF, args = np.array([1,]),alp = 0.5, Nits = 10,fl = False):
    Cost = np.full(Nits,0.)
    dt = 0.02
    dcnt = 0
    for j in range(Nits):
        Cost[j], grd = OF(S, *args)
        print(Cost[j])
        print(dt)
        print(dcnt)
        #Cost[j] = S.Ftot
        if j > 0 and Cost[j-1] <  Cost[j]:
            dt *= 1/3.
            dcnt = 0
        else: 
            dcnt += 1
        if dcnt > 5:
            dt *= 2.
            dcnt = 0
        if j > -1:
            step = np.cross(S.m,grd,axis=0)
            step += alp * np.cross(S.m,step,axis=0)
        else:
            step[:] = np.cross(S.m,grd + alp * step,axis=0)
        
        if fl:
            stepdir = 0.5*step / np.sum(step**2,axis=0)**0.5
            S.m[:] = stepdir[0] * (np.roll(S.m,-1,axis=1)+np.roll(S.m,1,axis=1)) \
                + stepdir[1] * (np.roll(S.m,-1,axis=2)+np.roll(S.m,1,axis=2)) \
                    +stepdir[2] * (np.roll(S.m,-1,axis=3)+np.roll(S.m,1,axis=3))
        S.m +=  - dt * step
        S.m *= 1/ np.sqrt(np.sum(S.m**2,axis=0))
    #S.FGrad()
    #Cost[Nits] = S.Ftot
    
    return Cost

def Recon(S, OF,  args = np.array([1,]), SP = None, flnm = None, Nits = 5, Nlam = 2, ld = False, prg = False):
    
    
     
    
    Cost = np.full(Nits + 1,0.)
    
    if ld:
        print('Computing Cost Derivatives...')
    
    S.Angular()
    
    Cost[0], gm = OF(S, *args) #X.dchidm + Boltz * S.dFdm #- lm * (S.m - v) #Boltz * dFdm + field[:,None,None,None] #+ DOF * (M0 - M0g)**2 /M0g**2
    
    gTH, gPH = S.AngularGrad(gm)
    
    gTHm1= 0. + gTH
    gPHm1 = 0. + gPH
    
    hTH = (0. + gTH).astype(np.float32)
    hPH = (0. + gPH).astype(np.float32)
    
    if flnm != None:
        SP(flnm, S, Cost, *args, init = True)
    
    
    #Cost[0] = Boltz*F-np.sum(field[:,None,None,None]*m)
    if ld:
        print('Initial Guess:')
        print('Total Cost: ' + str(Cost[0]))
        print()
        print('#############################')
    
    #return lam0, TH, PH, M0, hTH, hPH, Kappa, Boltz, field, dchidm, dchidM0, dFdm, dm, m, Cost, H0, fIm, Ikt, Akt, Projns, wts, I0, I00, N, q, Y, YT, W, WT, fftm, ifftm, a, fa, fftup, ifftup, aup, faup, spct, Nlam
    lam0 = np.full(2,0.)
    if prg == True:
        rng = tqdm(range(Nits))
    else:
        rng = range(Nits)
    for i in rng:
        
        #m += (np.random.random(m.shape)-0.5)*0.1/2**i
        #m *= 1/ np.sqrt(m[0]**2 + m[1]**2 + m[2]**2)[None,:]
        
        #lam0 = 0.1
        #lam0, TH, PH, M0, hTH, hPH, Kappa, Boltz, field, dchidm, dchidM0, dFdm, dm, m, Cost, H0, fIm, Ikt, Akt, Projns, wts, I0, I00, N, q, Y, YT, W, WT, fftm, ifftm, a, fa, fftup, ifftup, aup, faup, spct, Nlam
        Cost[i+1] = LMM(S, OF, args, gm, hTH, hPH, Cost[i], Nlam, ld, lam0)
        
        
        if ld:
            print('Step ' + str(i+1) + ' of ' + str(Nits) + ' complete!')
            print('Total Cost: ' + str(Cost[i+1]))
            print()
            print('#############################')
        
        if flnm != None:
            SP(flnm, S, Cost, *args)
        
        #gm = X.dchidm + Boltz * S.dFdm #+ field[:,None,None,None] #+ DOF * (M0 - M0g)**2/M0g**2
        
        gTH, gPH = S.AngularGrad(gm)
        
        gamTHHS = np.sum(gTH * (gTHm1 - gTH)) / (1e-12 + np.sum(hTH * (gTHm1 - gTH)))
        gamTHDY = np.sum(gTH**2) / (1e-12 + np.sum(hTH * (gTHm1 - gTH)))
        gamPHHS = np.sum(gPH * (gPHm1 - gPH)) /(1e-12 +  np.sum(hPH * (gPHm1 - gPH)))
        gamPHDY = np.sum(gPH**2) / (1e-12 + np.sum(hPH * (gPHm1 - gPH)))
        
        gamTH = np.maximum(0,np.minimum(gamTHHS,gamTHDY))
        gamPH = np.maximum(0,np.minimum(gamPHHS,gamPHDY))
        
    
        hTH = gTH + gamTH * hTH
        hPH = gPH + gamPH * hPH
        #hM0 = gM0 + gamM0 * hM0
        
        gTHm1 = 0. + gTH
        gPHm1 = 0. + gPH
        #gM0m1 = 0. + gM0
        
        
    
        
    return Cost

def LMM(S, OF, args, gm, hTH, hPH, Costm1, Nlam, ld, lam0):
    
    '''
    Solves for Step size in TH, PH, and M0 along hTH, hPH, and hM0
    
    rewrites dchidm, dFdm, m to their optimal values 
    
    returns optimized Cost and predicted SANS image
    
    '''
    
    # Initialize while loop condition and counter
    dCost = 10.
    ct = 0
    
    # Derivative and Hessian Matrices
    lam = np.full(2,0.)
    
    beta = S.CalcBetaAngular(gm, hTH, hPH)
    
    if lam0[0] == 0:
        lam = 0.01*np.abs(Costm1) / beta #* beta / (np.sum(beta**2))**0.5 
    else:
        lam = 0.1 * lam0 #*( 0. * beta + 1)
        lam0 *= 0.
    #lam[1] = 1e-3
    H = 0. * np.identity(2)
    
    Cost = Costm1 + 0.
    Costsv = Cost + 0.
    gmsv = 0. + gm
    resets = 0
    resetmax = 3
    while dCost > 1:
        
        #### Compute Alpha and Beta ####
        #### Function of TH, PH, M0, dchidm, dFdm, dm, m ###
        ### Compute lambda #
        
        
        '''
        alphap = alpha * (1 + lam0*np.identity(2))
        
        lam = np.linalg.lstsq(alphap,beta,rcond=None)[0]
        print(lam)
        '''
        if ld:
            print('Step Size: ' + str(lam))
            print('dchidlam: ' + str(beta))
            print()
        S.TH += lam[0]*hTH
        S.PH += lam[1]*hPH
        
        S.m[1] = np.sin(S.TH) * np.cos(S.PH)
        S.m[2] = np.sin(S.TH) * np.sin(S.PH)
        S.m[0] = np.cos(S.TH)
        
        Cost, gm[:] = OF(S, *args)
        
        if ld:
            print('Step size iteration ' + str(ct + 1) + ' of a possible ' + str(Nlam) + ' complete!')
            print('Total Cost: ' + str(Cost))
        
        
        if Cost >= Costm1:
            
            #lam0 = 1e3*lam0
            
            ## Revert Costly variables ###
            
            S.TH += - lam[0]*hTH
            S.PH += - lam[1]*hPH
            #lam0 += -lam
            lam *= 0.1
            
            #TH += - lam[0]*hTH
            #PH += - lam[1]*hPH
            
            #M0 += - lam[2]*hM0
            
            S.m[1] = np.sin(S.TH) * np.cos(S.PH)
            S.m[2] = np.sin(S.TH) * np.sin(S.PH)
            S.m[0] = np.cos(S.TH)
            
            
            if Nlam == 2 and resets < resetmax:
                ct += -2
                resets += 1
                if ld:
                    if resets == resetmax:
                        print('I reached the max number of resets, ' + str(resetmax) + ', and have to move on.')    
                    else:
                        print('Failed Step: I need to run extra iterations!')
        
        else:
            lam0 += lam

            betam1 = 0. + beta
            beta = S.CalcBetaAngular(gm, hTH, hPH)
            
            
            if ct > 5:
                dCost = Costm1 - Cost
            #print(dchisq)
            Costm1 = 0. + Cost
            bHb = np.sum((betam1-beta)[None,:]*H*(betam1-beta)[:,None])
            if np.all(bHb == 0):
                H += lam[:,None] * lam[None,:] / (1e-12 + np.sum(lam * (betam1 - beta))) 
                      
            else:
                u = lam / np.sum(lam * (betam1 - beta)) - H.dot(betam1 - beta) / bHb
                H += ( lam[:,None] * lam[None,:] / np.sum(lam * (betam1 - beta)) 
                      - H.dot(betam1 - beta)[:,None] * H.dot(beta-betam1)[None,:] / bHb
                      + bHb*u[:,None]*u[None,:]
                      )
            lam = H.dot(beta)
            
            Costsv = 0. + Cost
            gmsv = 0. + gm
        #print(dCost)
        ct += 1
        ct = np.maximum(ct,0)
        
        if ct > Nlam - 1:
            ct += -1
            dCost = 0
    
    if ld:
        print("Total Step " + str(lam0))
    ct += 1
    gm[:] = gmsv
    
    return Costsv





