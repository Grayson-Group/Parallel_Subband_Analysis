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
    
    
    Vg_val = 600
    Rxx = 1         ###1 or 2, selects whether to use Rxx_x (1) or Rxx_x2 (2)
    Rotate_list = [10, 11.5, 12.1]

    ### Vg vals where lockin2XX should be True: 
    lockin4_Vgs = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
    ### Vg vals where lockin2XX should be False:
    lockin2_Vgs = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]

     

    #Handle whether lockin_2 is measuring Rxx or Rxy
    if (Vg_val in lockin2_Vgs) & (Vg_val in lockin4_Vgs):
        lockin2xx_bool = False    #User defined default choice of lockin2xx_bool if gate voltage occurs in both arrays
    elif Vg_val in lockin4_Vgs:
        lockin2xx_bool = True       #If Vg_val only occurs in lockin2_Vgs, then lockin2 measures Rxx
    elif Vg_val in lockin2_Vgs:
        lockin2xx_bool = False      #If Vg_val only occurs in lockin2_Vgs, then lockin3 measures Rxx
    

    #### Run ParallelAnalysis with input Vg and neccessary lockin2xx bool and Rotate list  ####
    
    # inv, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
    #                                        B_start = 0, B_end = 1.5)
    inv, FFT, Rxx_grad, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                           B_start = 0.1, B_end = 1.5)
    
    #TO DO: Add contour plot function when an array of gate voltages is passed
    #NOTE: Least blind interpolation TO DO: Interpolate so that all data is equally spaced in B
        #Note: Should probably add all important FFT data to the inv dataframe, since that is returned to Main.py
    #Check to see if there is any offset of B = 0 (see symmetric B field sweep graph)
        #Artifically shift all data by some amount delta B (NOT 1/B), can we cause oscillations to become better/worse?
        #This would be caused by some background polarization of magnet when zero current is incident
        #Plot -B and +B data on top of each other from Christian data to determine B offset


    #Fourier transform raw 1/B data, delete peaks in FFT, then inverse fourier transform the remaining noise
    #to determine what features are causing this nosie.
    if 1 == 0:
        i = 0
        f_data = np.zeros([len(Vg_val), 1])
        spect_data = np.zeros([len(Vg_val), 1])

        for GV in Vg_val:
            print(GV)


            #Handle whether lockin_2 is measuring Rxx or Rxy
            if (GV in lockin2_Vgs) & (GV in lockin4_Vgs):
                lockin2xx_bool = False    #User defined default choice of lockin2xx_bool if gate voltage occurs in both arrays
            elif GV in lockin4_Vgs:
                lockin2xx_bool = True       #If Vg_val only occurs in lockin2_Vgs, then lockin2 measures Rxx
            elif GV in lockin2_Vgs:
                lockin2xx_bool = False      #If Vg_val only occurs in lockin2_Vgs, then lockin3 measures Rxx
            
            #trans = 
            inv, FFT, Rxx_grad, nu_bounds = PSIA.ParallelAnalysis(Vg = GV, lockin2XX = lockin2xx_bool, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                                    B_start = 0.1, B_end = 3.0)


            fft_start = 0#3520
            fft_cutoff = -1#-3520
            if i == 0:
                R_data = np.empty((len(GV),len(inv.An_Field)))
                spect_data = np.empty((len(GV),len(FFT.f_array)))
                B_data = np.empty(len(inv.An_Field))
                R_data[i,:] = D230831B_6_data.Rxx_x#/np.max(D230831B_6_data.Rxx_x)
                B_data = inv.An_Field
                f_data = FFT.f_array
                spect_data[i,:] = np.abs(FFT.Trans)/np.amax(np.abs(FFT.Trans))
            else: 
                R_data[i,:] = np.interp(B_data[:],inv.An_Field,D230831B_6_data.Rxx_x)
                spect_data[i,:] = np.interp(f_data[:],FFT.f_array,np.abs(FFT.Trans)/np.amax(np.abs(FFT.Trans)))
            i+= 1


            
            f_data[i] = FFT["f_array"].values
            spect_data[i] = FFT["Trans"].values

            #x = f_data
            #y = gate voltages
            #z = spect_data (trans)


        #plt.contourf(1e-4*f_data,  Vg_vals, np.abs(spect_data),levels=level_array)
        plt.contourf(1e-4*f_data,  Vg_val, np.abs(spect_data))
        plt.title("TEST")

        
    
    plt.show()
    
    
    
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