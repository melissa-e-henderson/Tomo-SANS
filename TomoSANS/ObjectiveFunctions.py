# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:18:31 2022

@author: bjh3
"""

import numpy as np


def ObjFun(S, X, lm):
    
    X.ComputeGrad(S, ld = False)
    S.FGrad()
    
    return lm[0] * X.chisq + lm[1] * S.Ftot, lm[0] * X.dchidm + lm[1] * S.dFdm

def ObjFun2(S, X, lm):
    
    X.ComputeGrad(S, ld = False)
    S.FGrad()
    
    mnfld = np.sum((np.mean(S.m,axis=(1,2,3)) - lm[3])**2) 
    mnfldg = - 2 * (np.mean(S.m,axis=(1,2,3)) - lm[3])[:,None,None,None] 
    
    return lm[0] * X.chisq + lm[1] * S.Ftot + lm[2] * mnfld, lm[0] * X.dchidm + lm[1] * S.dFdm + lm[2] * mnfldg

def EMin(S, args):
    
    S.FGrad()
    
    return S.Ftot, S.dFdm

def EMinAve(S, args):
    
    S.FGrad()
    mnfld = np.sum((np.mean(S.m,axis=(1,2,3)) - args[1])**2)
    mnfldg = - 2 * (np.mean(S.m,axis=(1,2,3)) - args[1])[:,None,None,None] 
    
    
    return S.Ftot + args[0]*mnfld, S.dFdm + args[0]*mnfldg

def EMinxy(S, args):
    
    S.FGrad()
    
    xymsk = np.array([0,1,1])
    
    return S.Ftot, S.dFdm * xymsk[:,None,None,None]

def ProX(S, X, V, lm):
    
    X.ComputeGrad(S, ld = False)
    
    return lm[0] * X.chisq + 0.5*lm[1]*np.sum((S.m - V)**2), \
        lm[0] * X.dchidm - lm[1] * (S.m - V)
        
def ProF(S, V, lm):
    
    S.FGrad()
    
    return lm[0] * S.Ftot + 0.5*lm[1]*np.sum((S.m - V)**2), \
        lm[0] * S.dFdm - lm[1] * (S.m - V)

def ProXF(S, X, V, lm):
    
    X.ComputeGrad(S, ld = False)
    S.FGrad()
    
    return lm[0] * X.chisq + lm[1] * S.Ftot + 0.5*lm[2]*np.sum((S.m - V)**2), \
        lm[0] * X.dchidm + lm[1] * S.dFdm - lm[2] * (S.m - V)