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


#TO DO: ADD WATERFALL PLOT (3D Contour plot)

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
    Vg_val = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]                   
    #Vg_val = "D230831B_4_Last_"                       
#                                   List of Specialized Data Files 
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
            lockin2xx_bool = True   #All specialty files use lockin 2 to measure Rxx
        
        if type(Vg_val) == int:
            lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = Vg_val, default_bool = False)  #Determine if lockin2 measures Rxx or Rxy  
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
             
        inv, FFT, Rxx_input, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, lockin2XX = lockin2xx_bool, gradient = grad, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                            B_start = 0.1, B_end = 0.51)
            

    if type(Vg_val) == list:
        #Vg_val is an array of gate voltages
        #### Run ParallelAnalysis once for each value in Vg_val, create a contour plot  ####
        
        
        i = 0

        for GV in Vg_val:
            
            lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = GV, default_bool = True)
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
                
            inv, FFT, Rxx_input, nu_bounds = PSIA.ParallelAnalysis(Vg = GV, lockin2XX = lockin2xx_bool, gradient = grad, Rxx_1or2 = Rxx, I = 2e-6, Iscaler = 0.9701, Rotate = Rotate_list, ne = 4E15, 
                                                    B_start = 0.1, B_end = 0.51)



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
        ax1.set_title("Column (x) stride set to 0")
        ax1.set_xlabel("Carrier Conc. ($cm^{-2}$)")
        ax1.set_ylabel("Gate Voltage ($mV$)")
        ax1.set_zlabel("FFT Amplitude")
        ax1.set_xlim(0,5e11)
        
        # Give the second plot only wireframes of the type x = c
        #ax2.plot_wireframe(1e-4*f_data,  Vg_val, np.abs(spect_data), rstride=0, cstride=10)
        #ax2.set_title("Row (y) stride set to 0")
        
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