# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:36:14 2023

@author: Madma
"""
import scipy.constants as c
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.figure as fig
import numpy as np
from scipy import fft as ft
import D181211A1_QHE_Fourier_Analysis as QFT
import Parallel_Subband_Inversion_Analysis as PSIA
from mpl_toolkits.mplot3d import axes3d

######
#TO DO:
    #Add contour plot function when an array of gate voltages is passed
    #Check to see if there is any offset of B = 0 (see symmetric B field sweep graph)
        #Artifically shift all data by some amount delta B (NOT 1/B), can we cause oscillations to become better/worse?
        #This would be caused by some background polarization of magnet when zero current is incident
        #Plot -B and +B data on top of each other from Christian data to determine B offset


    #######FIX _3 CONTACT PLOT
    #They are all differ Rxx configurations, need to label plots as such



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
    #Vg_val = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]  
    #Vg_val = 300
    Vg_val = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
                 
    #Vg_val = "D230831B_4_Last_"                       
#                                   List of Specialized Data Files 
                        #D230831B_3_Contacts_000mV_Vg.dat                  Vg_val = "D230831B_3_Contacts_"
                        #D230831B_4_LowField_000mV_Vg.dat                  Vg_val = "D230831B_4_LowField_"
                        #D230831B_4_Last_000mV_Vg.dat                      Vg_val = "D230831B_4_Last_"
                        #D230831B_4_Negative_200mV_Vg.dat                  Vg_val = "D230831B_4_Negative_"
                        ####NOTE: All specialized data files have lockin2xx_bool == True

    
    B_range = [0.1, 0.45]  #2 ELEMENT ARRAY: Range of B field (Tesla) to take FFT of
    
    
    Rxx = 2         ###1 or 2, selects whether to use Rxx_x (1) or Rxx_x2 (2) for any FFT analysis
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
            lockin2xx_bool = True   #All specialty files use lockin 2 to measure Rxx
        
        if type(Vg_val) == int:
            lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = Vg_val, default_bool = True)  #Determine if lockin2 measures Rxx or Rxy  
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
             
        inv, FFT, Rxx_input, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, gradient = grad, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                            B_start = B_range[0], B_end = B_range[1])
            

    if type(Vg_val) == list:
        #Vg_val is an array of gate voltages
        #### Run ParallelAnalysis once for each value in Vg_val, create a contour plot  ####
        
        
        i = 0

        for GV in Vg_val:
            
            lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = GV, default_bool = True)
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
                
            inv, FFT, Rxx_input, nu_bounds = PSIA.ParallelAnalysis(Vg = GV, lockin2XX = lockin2xx_bool, gradient = grad, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                                    B_start = B_range[0], B_end = B_range[1])



            #TO DO: Add fft_start and fft_cutoff parameters to only FFT a given slice of data
            #fft_start = 0
            #fft_cutoff = -1
            
            if i == 0:
                
                Trans = np.empty((len(Vg_val), len(FFT.f_array)), dtype = np.complex128)
                f_array = np.empty((len(Vg_val), len(FFT.f_array)))
                gate_val = np.empty((len(Vg_val), len(FFT.f_array)))
                
                Trans[i,:] = FFT.Trans                
                f_array[i,:] = FFT.f_array
                gate_val[i,:].fill(Vg_val[i])
                #R_data = np.empty((len(Vg_val),len(inv.B_field)))
                #f_data = np.empty((len(Vg_val), len(FFT.f_array)))
                #spect_data = np.empty((len(Vg_val),len(FFT.f_array)))
                #B_data = np.empty(len(inv.B_field))
                #R_data[i,:] = inv.Rxx#/np.max(D230831B_6_data.Rxx_x)
                #f_data[i,:] = FFT.f_array
                #B_data = inv.B_field
                #spect_data[i,:] = np.abs(FFT.Trans)/np.amax(np.abs(FFT.Trans))
            else: 
                Trans[i,:] = FFT.Trans                
                f_array[i,:] = FFT.f_array
                gate_val[i,:].fill(Vg_val[i])

                
                # R_data[i,:] = np.interp(B_data[:],inv.B_field, inv.Rxx)
                #spect_data[i,:] = np.interp(f_data[:],FFT.f_array,np.abs(FFT.Trans)/np.amax(np.abs(FFT.Trans)))
                #f_data[i,:] = FFT.f_array

            i+= 1
            
            
            
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

        ax1.plot_wireframe(f_array*(2*c.e/c.h)*1e-4,  gate_val, np.abs(Trans), rstride=1, cstride=0)
        ax1.set_title("FFT results from {}mV to {}mV".format(Vg_val[0], Vg_val[-1]))
        ax1.set_xlabel("Carrier Conc. ($cm^{-2}$)")
        ax1.set_ylabel("Gate Voltage ($mV$)")
        ax1.set_zlabel("FFT Amplitude")
        ax1.set_xlim(0,5e11)
        
        
        plt.tight_layout()
        
        
        
        '''
        plt.figure()
        plt.contourf(1e-4*f_data,  Vg_val, np.abs(spect_data))
        plt.xlim([0,5e11])
        plt.xlabel("Carrier Concentration $(cm^{-2})$")
        plt.ylabel("Gate Volage $(mV)$")
        plt.annotate(text=r"$B$ range = ["+ np.format_float_positional(np.round(np.min(inv.B_field), 1), unique = False, precision=1)+ r" T, "+np.format_float_positional(np.round(np.max(inv.B_field), 1), unique = False, precision=1)+r"T]",
                     xy=[0.65,0.95],
                     xycoords='axes fraction')
        plt.title("Gate Voltage, 1/B FFT, and FFT intensity ")

        '''
        
    
    plt.show()
    
    
    
    
    
    
    
    
    '''Plot Peaks of FFT vs. Gate Volt'''
    
    
    
    #######  D230831B_2_,          Rxx = 1   #########################
    
    GV = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]

    Const = [0.138, 0.16, 0.16, 0.14, 0.16, 0.16, 0.2, 0.12, 0.14, 0.34, 0.16, 0.16, 0.2, 0.16, 0.14, 0.14, 0.12, 0.14]
    B1 = [0.16, 0.46, 0.56, 0.71, 0.87]
    B2 = [3.81, 3.89, 3.81, 3.85, 3.91, 3.81, 3.85, 3.89, 3.93, 3.87, 3.95, 3.91, 3.95, 3.83, 3.93, 3.81, 3.83]
    B3 = [3.997, 4.25, 4.52, 4.6, 4.7, 4.82, 4.90, 5.02, 5.10, 5.22, 5.28, 5.42, 5.4, 5.48, 5.77, 5.65, 5.77, 5.93]
    
    
    
    #######  D230831B_2_,          Rxx = 2   #########################
    
    GV = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]
    
    Const = [0.16, 0.16, 0.16, 0.14, 0.1, 0.16, 0.14, 0.16, 0.14, 0.12, 0.3, 0.16, 0.22, 0.18, 0.16, 0.14, 0.16, 0.18]
    B1 = [0.46, 0.6, 0.77, 0.85]
    B2 = [3.87, 3.87, 3.95, 3.89, 3.87, 3.83, 3.85, 3.86, 3.95, 3.93, 3.97, 3.95, 3.95, 3.99, 3.93, 3.97, 3.97]
    B3 = [3.97, 4.21, 4.5, 4.58, 4.74, 4.86, 4.88, 5.04, 5.08, 5.20, 5.22, 5.38, 5.1, 5.6, 5.87, 6.05, 6.29, 6.61]
    Mystery = [4.23, 4.19, 4.11, 4.05, 4.05, 3.95, 4.25, 4.25, 4.33, 4.31, 4.40, 4.48, 4.48, 4.52, 4.66]
    
    
    
    #######  D230831B_4_,          Rxx = 1   #########################
    
    GV = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
    
    Const = [0.15, 0.16, 0.14, 0.16, 0.2, 0.22, 0.24, 0.12, 0.14, 0.12, 0.14, 0.14]
    B1 = [0.22, 0.24, 0.48, 0.52, 0.75, 0.79, 0.79]
    B2 = [4.09, 3.93, 3.93, 3.95, 3.92, 3.89, 3.87, 3.89, 3.89, 3.77, 3.97, 3.95]
    B3 = [4.09, 4.52, 4.82, 4.9, 4.98, 5.3, 5.34, 5.34, 5.38, 5.58, 5.65, 5.65]
    
    
    #######  D230831B_4_,          Rxx = 2   #########################
    
    GV = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
    
    Const = [0.18, 0.14, 0.12, 0.12, 0.12, 0.16, 0.14, 0.14, 0.12, 0.12, 0.16, 0.16]
    B1 = [0.24, 0.52, 0.56, 0.77, 0.81, 0.85]
    B2 = [3.88, 3.93, 3.95, 3.91, 3.99, 3.97, 3.97, 3.95, 3.97, 3.97, 3.99]   
    B3 = [4.09, 4.56, 4.88, 4.94, 5.00, 5.28, 5.28, 5.36, 5.36, 5.56, 5.60, 5.62]
    Mystery = [4.13, 4.27, 4.21, 4.07, 4.25, 4.35, 4.38, 4.44, 4.48, 4.72, 4.7]
    
    
    
    
    
    graph, ax1 = plt.subplots()    
    ax1.plot(Const, GV, c = 'r')
    ax1.plot(B1, GV[-len(B1):], c = 'r', label = "$1^{st}$ Peak")
    ax1.plot(B2, GV[-len(B2):], label = "$2^{nd}$ Peak")
    ax1.plot(B3, GV, label = "$3^{rd}$ Peak")
    
    ax1.plot(Mystery, GV[-len(Mystery):], label = "Mystery")
    
    ax1.set_ylabel("Gave Voltage ($mV$)")
    ax1.set_xlabel("Magnetic Field ($B$)")
    ax1.set_title("Peak locations in FFT results vs. Gate voltage")
    
    secax = ax1.secondary_xaxis('top', functions=(QFT.B_to_n, QFT.n_to_B))
    secax.set_xlabel("Carrier Conc. ($cm^{-2}$)")
    
    ax1.annotate(text=r"$B$ range of FFT = ["+ np.format_float_positional(B_range[0], unique = False, precision=2)+ r" T, "+np.format_float_positional(B_range[1], unique = False, precision=2)+r"T]",
             xy=[0.1,0.2],
             xycoords='axes fraction')
    if lockin2xx_bool == True:
        if Rxx == 1:
            ax1.annotate(text="Rxx_1, D230831B_4_", xy=[0.1,0.15], xycoords='axes fraction')
        elif Rxx == 2:
            ax1.annotate(text="Rxx_2, D230831B_4_", xy=[0.1,0.15], xycoords='axes fraction')
    else:
        if Rxx == 1:
            ax1.annotate(text="Rxx_1, D230831B_2_", xy=[0.1,0.15], xycoords='axes fraction')
        elif Rxx == 2:
            ax1.annotate(text="Rxx_2, D230831B_2_", xy=[0.1,0.15], xycoords='axes fraction')
    
    ax1.grid()
    ax1.legend()
    
    
    
    
    
    
    
    
    
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