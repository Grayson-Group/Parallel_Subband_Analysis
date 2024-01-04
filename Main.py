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
    #Add contour plot function when an array of gate voltages is passed
    #NOTE: Least blind interpolation TO DO: Interpolate so that all data is equally spaced in B
        #Note: Should probably add all important FFT data to the inv dataframe, since that is returned to Main.py
    #Check to see if there is any offset of B = 0 (see symmetric B field sweep graph)
        #Artifically shift all data by some amount delta B (NOT 1/B), can we cause oscillations to become better/worse?
        #This would be caused by some background polarization of magnet when zero current is incident
        #Plot -B and +B data on top of each other from Christian data to determine B offset


    #Fourier transform raw 1/B data, delete peaks in FFT, then inverse fourier transform the remaining noise
    #to determine what features are causing this nosie.



    #All in 1/B, post derivative data
    #average value of derivative might not be zero
    #Take left side and right side, estimate average on each side, 

    #Take average of integer number of oscillations on left or right
    #Subtract away a line connecting these two averages
    #DO THIS BEFORE APODIZATION
    #(See line 212 in QFT)


    ###TO DO:  Plot real and imaginary part of FFT, not just amplitude


'''
###########################
USE THESE PARAMETERS:
  I = 2e-6
  Iscalar = 0.97
  Rotate = [10, 11.5, 12.1]   ([Lockin_1 phase, Lockin_2 phase, Lockin_3 phase])
    
###########################
'''


if __name__ == "__main__":
    
    Von_Klitz = 25812.80745
    
##################################################################################    
                                  #GATE VALUE(s) THAT YOU WANT TO ANALYSE
                        #Should be an int that is an element of lockin4_Vgs or lockin2_Vgs
                        #OR it can be a list of int which are elements of lockin4_Vgs or lockin2_Vgs
                        #OR it can be a string for specialized data files
    Vg_val = 000                   
    #Vg_val = "D230831B_4_Last_"                       
#                                   List of Specialized Data File Vg_val
                        #D230831B_3_Contacts_000mV_Vg.dat                  Vg_val = "D230831B_3_Contacts_"
                        #D230831B_4_LowField_000mV_Vg.dat                  Vg_val = "D230831B_4_LowField_"
                        #D230831B_4_Last_000mV_Vg.dat                      Vg_val = "D230831B_4_Last_"
                        #D230831B_4_Negative_200mV_Vg.dat                  Vg_val = "D230831B_4_Negative_"
                        ####NOTE: All specialized data files have lockin2xx_bool == True


    Rxx = 1         ###1 or 2, selects whether to use Rxx_x (1) or Rxx_x2 (2) for any FFT analysis
    grad = False    ###If true, FFT will be calculated using DERIVATIVE of Rxx vs. 1/B. If false use raw Rxx vs. 1/B
    
###################################################################################    
    
    Rotate_list = [10, 11.5, 12.1]



    ### Vg vals where lockin2XX should be True: 
    lockin4_Vgs = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
    ### Vg vals where lockin2XX should be False:
    lockin2_Vgs = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]

    



    if type(Vg_val) == int or type(Vg_val) == str:
    #### Run ParallelAnalysis with single input Vg_val ####    
        
        if type(Vg_val) == str:
            lockin2xx_bool = True
        
        if type(Vg_val) == int:
            lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = Vg_val, default_bool = True)  #Determine if lockin2 measures Rxx or Rxy  
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
             
        inv, FFT, Rxx_grad, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, gradient = grad, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                            B_start = 0.1, B_end = 0.51)
            

    if type(Vg_val) == list:
        #Vg_val is an array of gate voltages
        #### Run ParallelAnalysis once for each value in Vg_val, create a contour plot  ####
        
        
        i = 0

        for GV in Vg_val:
            
            lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = GV, default_bool = True)
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
                
            inv, FFT, Rxx_grad, nu_bounds = PSIA.ParallelAnalysis(Vg = GV, lockin2XX = lockin2xx_bool, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                                    B_start = 0.1, B_end = 3.0)



            #TO DO: Add fft_start and fft_cutoff parameters to only FFT a given slice of data
            #fft_start = 0
            #fft_cutoff = -1
            
            if i == 0:
                R_data = np.empty((len(Vg_val),len(inv.B_field)))
                spect_data = np.empty((len(Vg_val),len(FFT.f_array)))
                B_data = np.empty(len(inv.B_field))
                R_data[i,:] = inv.Rxx#/np.max(D230831B_6_data.Rxx_x)
                B_data = inv.B_field
                f_data = FFT.f_array
                spect_data[i,:] = np.abs(FFT.Trans)/np.amax(np.abs(FFT.Trans))
            else: 
                R_data[i,:] = np.interp(B_data[:],inv.B_field, inv.Rxx)
                spect_data[i,:] = np.interp(f_data[:],FFT.f_array,np.abs(FFT.Trans)/np.amax(np.abs(FFT.Trans)))
            i+= 1

        plt.figure()
        plt.contourf(1e-4*f_data,  Vg_val, np.abs(spect_data))
        plt.xlim([0,5e11])
        plt.xlabel("Carrier Concentration $(cm^{-2})$")
        plt.ylabel("Gate Volage $(mV)$")
        plt.annotate(text=r"$B$ range = ["+ np.format_float_positional(np.round(np.min(inv.B_field), 1), unique = False, precision=1)+ r" T, "+np.format_float_positional(np.round(np.max(inv.B_field), 1), unique = False, precision=1)+r"T]",
                     xy=[0.65,0.95],
                     xycoords='axes fraction')
        plt.title("Gate Voltage, 1/B FFT, and FFT intensity ")

        
        
    
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