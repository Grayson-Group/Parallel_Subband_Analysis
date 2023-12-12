# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:36:14 2023

@author: Madma
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import numpy as np
from scipy import optimize as opt
from scipy import stats as st
from scipy import fft as ft
from scipy import signal as sig
import scipy.constants as c
import os
from operator import index, indexOf
import D181211A1_QHE_Fourier_Analysis as QFT
import Parallel_Subband_Inversion_Analysis as PSIA

######
#TO DO:
    #Plot all gate voltages as 1/B
    #At low B field (shubnikov de haas) try to identify w (freq) of both subbands
    #Plot frequency of both subbands as a function of gate voltage
    
    #Add option to FFT R_xx or R_xx2 data for any datafile
    
###########################
#USE THESE PARAMETERS for ParallelAnalysis:
#  I = 2e-6
#  Iscalar = 0.97
#  Rotate = [10, 11.5, 12.1]
#    
###########################



if __name__ == "__main__":
    
    Von_Klitz = 25812.80745
    

    Vg_val = 000


    ### Vg vals where lockin2XX should be True:
    lockin2_Vgs = [000, 100, 250]

    ### Vg vals where lockin2XX should be False:
    lockin4_Vgs = [200]

    Rotate_list = [10, 11.5, 12.1]

    if Vg_val in lockin2_Vgs:
        lockin2xx_bool = True
    elif Vg_val in lockin4_Vgs:
        lockin2xx_bool = False

    ### Run ParallelAnalysis with input Vg and neccessary lockin2xx bool and Rotate list
    
    # inv, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
    #                                        B_start = 0, B_end = 1.5)
    inv, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                           B_start = 0.1, B_end = 3.0)
    

    # inv, nu_bounds = PSIA.ParallelAnalysis(Vg = 000, lockin2XX = False, I = 2e-6, Iscaler = 0.9707, Rotate = [10, 12.1, 11.5], ne = 4E15)
    # inv, nu_bounds = PSIA.ParallelAnalysis(Vg = 100, lockin2XX = True, I = 2e-6, Iscaler = 0.9701, Rotate = [10, 11.5, 12.1], ne = 4E15)
    # inv, nu_bounds = PSIA.ParallelAnalysis(Vg = 200, lockin2XX = False, I = 2e-6, Iscaler = 0.9701, Rotate = [10, 11.5, 12.1], ne = 4E15)
    
    # inv2, nu_bounds2 = PSIA.ParallelAnalysis(Vg = 100, lockin2XX = True, I = 2e-6, Iscaler = 0.9701, Rotate = [10, 11.5, 12.1], ne = 4E15)
    
    plt.show()
    
    # new_bounds = PSIA.scaling(inv, inv2, 0.3, nu_bounds[1], False)
    
    
    #plt.plot(inv2.B_field[new_bounds[0]:new_bounds[1]], inv2.Rxx[new_bounds[0]:new_bounds[1]])
    
    
    
    '''
    PLATEAU COMPARE
    
    
    plateau_1 = np.average(inv["Rxy"][nu_bounds[1][0]:nu_bounds[1][1]])
    plateau_2 = np.average(inv["Rxy"][nu_bounds[2][0]:nu_bounds[2][1]])
    plateau_3 = np.average(inv["Rxy"][nu_bounds[3][0]:nu_bounds[3][1]])
    plateau_4 = np.average(inv["Rxy"][nu_bounds[4][0]:nu_bounds[4][1]])
    
    
    print(plateau_1 / plateau_2)
    print(plateau_1 / plateau_3)
    print(plateau_1 / plateau_4)
    print(plateau_2 / plateau_4)
    
    
    print(plateau_1 - (Von_Klitz/1))
    print(plateau_2 - (Von_Klitz/2))
    print(plateau_3 - (Von_Klitz/3))
    print(plateau_4 - (Von_Klitz/4))
    '''
    
    
    