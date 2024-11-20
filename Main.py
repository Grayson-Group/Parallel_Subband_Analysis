# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:36:14 2023

@author: Madma
"""
#BOO
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
USE THESE PARAMETERS for ETH data:
  I = 2e-6
  Iscalar = 0.97
  Rotate = [10, 11.5, 12.1]   ([Lockin_1 phase, Lockin_2 phase, Lockin_3 phase])
    
###########################
'''


if __name__ == "__main__":
    
    Von_Klitz = 25812.80745
    NU = True    #If true, will use Northwestern University data architecture. If false, use ETH's data architecture
    
    
    if NU == False: 
                                    ####IF USING ETH DATA####
##################################################################################    
                                  #GATE VALUE(s) THAT YOU WANT TO ANALYSE
                        #Should be an int that is an element of lockin4_Vgs or lockin2_Vgs
                        #OR it can be a list of int which are elements of lockin4_Vgs or lockin2_Vgs
                        #OR it can be a string for specialized data files
    
        #Vg_val = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]  
        #Vg_val = 000
        Vg_val = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
                         
        #Vg_val = "D230831B_4_Last_"                       
    #                                   List of Specialized Data Files 
                            #D230831B_3_Contacts_000mV_Vg.dat                  Vg_val = "D230831B_3_Contacts_"
                            #D230831B_4_LowField_000mV_Vg.dat                  Vg_val = "D230831B_4_LowField_"
                            #D230831B_4_Last_000mV_Vg.dat                      Vg_val = "D230831B_4_Last_"
                            #D230831B_4_Negative_200mV_Vg.dat                  Vg_val = "D230831B_4_Negative_"
                            ####NOTE: All specialized data files have lockin2xx_bool == True
        #Vg_val = "D230831B_4_LowField_"
        
        
        
        
        default_bool = True     #The default value for lockin2xx_bool, True of Rxx2 is measured by lockin 2
                                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
        
        
        I = 2e-6
        Iscalar = 0.97
        Rxx = 1        ###1 or 2, selects whether to use Rxx_x (1) or Rxx_x2 (2) for any FFT analysis

        Rotate_list = [10, 11.5, 12.1]
    
        B_range = [0.1, 0.45]

    
        ### Vg vals where lockin2XX should be True: 
        lockin4_Vgs = [000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650]
        ### Vg vals where lockin2XX should be False:
        lockin2_Vgs = [000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600]

    


    if NU == True:
                            ####IF USING NU DATA####
##################################################################################    


#240501_062_GaAs_D230831B_5_I100nA_G00_T2_B1_Vxx_Vxy
        NU_files = {0: "240501_062_GaAs_D230831B_5_I100nA_G00_T2_B1_Vxx_Vxy", 
                    100 : "240501_063_GaAs_D230831B_5_I100nA_G01_T2_B0_Vxx_Vxy",
                    200: "240501_064_GaAs_D230831B_5_I100nA_G02_T2_B1_Vxx_Vxy",
                    250: "240501_065_GaAs_D230831B_5_I100nA_G025_T2_B0_Vxx_Vxy",
                    300: "240501_066_GaAs_D230831B_5_I100nA_G03_T2_B1_Vxx_Vxy",
                    350: "240501_067_GaAs_D230831B_5_I100nA_G035_T2_B0_Vxx_Vxy",
                    400: "240501_068_GaAs_D230831B_5_I100nA_G04_T2_B1_Vxx_Vxy",
                    450: "240501_069_GaAs_D230831B_5_I100nA_G045_T2_B0_Vxx_Vxy",
                    500: "240501_070_GaAs_D230831B_5_I100nA_G05_T2_B1_Vxx_Vxy",
                    550: "240501_071_GaAs_D230831B_5_I100nA_G055_T2_B0_Vxx_Vxy",
                    600: "240501_072_GaAs_D230831B_5_I100nA_G06_T2_B1_Vxx_Vxy",
                    650: "240501_077_GaAs_D230831B_5_I100nA_G065_T2_B0_Vxx_Vxy",
                    700: "240501_078_GaAs_D230831B_5_I100nA_G07_T2_B1_Vxx_Vxy"}
        
        lockin2xx_bool = False
        Rotate_list = [0, 0, 0]

        I = 0.98E-7
        Iscalar = 1.0
        Rxx = 1  #This value is meaningless for NU data

        #Vg_val = 250
        Vg_val = [0, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        #Vg_val = 700

        
        
        B_range = [0.2, 0.85]
###################################################################################
    

   

    grad = 0    ###If true, FFT will be calculated using DERIVATIVE of Rxx vs. 1/B. If false use raw Rxx vs. 1/B
    if grad == 0:
        sigma = 0            ### True or False, do you want to perform FFT on the inverted CONDUCTIVITY_xx instead of R_xx
    



    
    if type(Vg_val) == int or type(Vg_val) == str:
    #### Run ParallelAnalysis with single input Vg_val ####    
        
        if NU == False:
            if type(Vg_val) == str:
                lockin2xx_bool = True   #All specialty files use lockin 2 to measure Rxx
            
            if type(Vg_val) == int:
                lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = Vg_val, default_bool = default_bool)  #Determine if lockin2 measures Rxx or Rxy  
                    #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                    #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
                 
        inv, FFT, Rxx_input, nu_bounds = PSIA.ParallelAnalysis(Vg = Vg_val, NU = NU, lockin2XX = lockin2xx_bool, gradient = grad, 
                                            Rxx_1or2 = Rxx, sigma = sigma, I = I, Iscalar = Iscalar, Rotate = Rotate_list, ne = 4E15, B_start = B_range[0], B_end = B_range[1])
            

    if type(Vg_val) == list:
        #Vg_val is an array of gate voltages
        #### Run ParallelAnalysis once for each value in Vg_val, create a contour plot  ####
        
        
        i = 0

        for GV in Vg_val:
            
            if NU == False:
                lockin2xx_bool = QFT.determine_Rxx_lockin(gate_val = GV, default_bool = default_bool)
                #default_bool = True will default grab data files ending in "_4_" as these files have lockin2 measuring Rxx
                #default_bool = False will default grab data files ending in "_2_" as these files have lockin2 measuring Rxy
                
            inv, FFT, Rxx_input, nu_bounds = PSIA.ParallelAnalysis(Vg = GV, NU = NU, lockin2XX = lockin2xx_bool, gradient = grad, sigma = sigma, Rxx_1or2 = Rxx, I = I, 
                                                    Iscalar = Iscalar, Rotate = Rotate_list, ne = 4E15, B_start = B_range[0], B_end = B_range[1])



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
    FFT_Peak_Anal = 1
    plot_fits = 1
    if FFT_Peak_Anal == True:
        
        if NU == False:
        
            if (default_bool == False and Rxx == 1 and sigma == 0):
                #######  D230831B_2_,          Rxx = 1   #########################
                
                GV = np.array([000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600])
            
                Const = [0.138, 0.16, 0.16, 0.14, 0.16, 0.16, 0.2, 0.12, 0.14, 0.20, 0.16, 0.16, 0.2, 0.16, 0.14, 0.14, 0.12, 0.14]
                B1 = [0.16, 0.46, 0.56, 0.71, 0.87]
                B2 = [3.81, 3.89, 3.81, 3.85, 3.91, 3.81, 3.85, 3.89, 3.93, 3.87, 3.95, 3.91, 3.95, 3.83, 3.93, 3.81, 3.83]
                B3 = [3.997, 4.25, 4.52, 4.6, 4.7, 4.82, 4.90, 5.02, 5.10, 5.22, 5.28, 5.42, 5.4, 5.48, 5.77, 5.65, 5.77, 5.93]
                
                
            
            
            if (default_bool == False and Rxx == 2 and sigma == 0):
            #######  D230831B_2_,          Rxx = 2   #########################
            
                GV = np.array([000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600])
                
                Const = [0.16, 0.16, 0.16, 0.14, 0.1, 0.16, 0.14, 0.16, 0.14, 0.12, 0.3, 0.16, 0.22, 0.18, 0.16, 0.14, 0.16, 0.18]
                B1 = [0.46, 0.6, 0.77, 0.85]
                B2 = [3.87, 3.87, 3.95, 3.89, 3.87, 3.83, 3.85, 3.86, 3.95, 3.93, 3.97, 3.95, 3.95, 3.99, 3.93, 3.97, 3.97]
                B3 = [3.97, 4.21, 4.5, 4.58, 4.74, 4.86, 4.88, 5.04, 5.08, 5.20, 5.22, 5.38, 5.1, 5.6, 5.87, 6.05, 6.29, 6.61]
                Mystery = [4.23, 4.19, 4.11, 4.05, 4.05, 3.95, 4.25, 4.25, 4.33, 4.31, 4.40, 4.48, 4.48, 4.52, 4.66]
                
                
            if (default_bool == True and Rxx == 1 and sigma == 0):
            #######  D230831B_4_,          Rxx = 1   #########################
                
                GV = np.array([000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650])
                
                Const = [0.15, 0.16, 0.14, 0.16, 0.2, 0.22, 0.24, 0.12, 0.14, 0.12, 0.14, 0.14]
                #B1 = [0.24, 0.48, 0.52, 0.75, 0.79, 0.79]
                #B2 = [3.93, 3.93, 3.95, 3.92, 3.89, 3.87, 3.89, 3.89, 3.77, 3.97, 3.95]
                #B3 = [4.09, 4.52, 4.82, 4.9, 4.98, 5.3, 5.34, 5.34, 5.38, 5.58, 5.65, 5.65]
                
                #B1 = [3.93, 3.93, 3.95, 3.92, 3.89, 3.87, 3.89, 3.89, 3.77, 3.97, 3.95]
                B1 = [4.09, 4.52, 4.82, 4.9, 4.98, 5.3]
                B2 = [5.3, 5.34, 5.34, 5.38, 5.58, 5.65, 5.65]
                B3 = [0.24, 0.48, 0.52, 0.75, 0.79, 0.79]

            
            if (default_bool == True and Rxx == 2 and sigma == 0):
            #######  D230831B_4_,          Rxx = 2   #########################
            
                GV = np.array([000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650])
                
                Const = [0.18, 0.14, 0.12, 0.12, 0.12, 0.16, 0.14, 0.14, 0.12, 0.12, 0.16, 0.16]
                B1 = [0.24, 0.52, 0.56, 0.77, 0.81, 0.85]
                B2 = [3.88, 3.93, 3.95, 3.91, 3.99, 3.97, 3.97, 3.95, 3.97, 3.97, 3.99]   
                B3 = [4.09, 4.56, 4.88, 4.94, 5.00, 5.28, 5.28, 5.36, 5.36, 5.56, 5.60, 5.62]
                Mystery = [4.13, 4.27, 4.21, 4.07, 4.25, 4.35, 4.38, 4.44, 4.48, 4.72, 4.7]
                
           
                
            if (default_bool == False and Rxx == 1 and sigma == 1):
               #######  D230831B_2_,          Rxx = 1         Conductivity     #########################
               
                GV = np.array([000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600])
           
                Const = [0.15, 0.16, 0.14, 0.12, 0.12, 0.14, 0.12, 0.14, 0.14, 0.14, 0.14, 0.16, 0.16, 0.1, 0.14, 0.12, 0.16, 0.16]
                B1 = [0.34, 0.4, 0.6, 0.69, 0.85]
                B2 = [3.8, 3.83, 3.93, 3.77, 3.83, 3.83, 3.95, 3.9, 3.95, 3.95, 3.93, 3.85, 3.93, 3.81, 3.85]
                B3 = [3.96, 4.21, 4.52, 4.6, 4.68, 4.86, 4.9, 5.02, 5.08, 5.2, 5.3, 5.42, 5.34, 5.48, 5.75, 5.69, 5.77, 5.91]
               
            
            if (default_bool == False and Rxx == 2 and sigma == 1):
               #######  D230831B_2_,          Rxx = 2      Conductivity  #########################
               
                GV = np.array([000, 100, 150, 175, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 450, 500, 550, 600])
           
                Const = [0.14, 0.14, 0.14, 0.14, 0.24, 0.34, 0.2, 0.42, 0.12, 0.12, 0.10, 0.14, 0.16, 0.1, 0.12, 0.12, 0.12, 0.12]
                B1 = [0.38, 0.42, 0.64, 0.77, 0.85]
                B2 = [3.89, 3.99, 4.09, 4.15, 4.11, 4.09, 3.95, 3.95, 3.89, 3.95, 3.93, 3.99, 3.95, 3.87, 3.91, 3.95]
                B3 = [3.95, 4.19, 4.5, 4.6, 4.72, 4.84, 4.86, 5.04, 5.06, 5.20, 5.24, 5.4, 5.16, 5.58, 5.38, 5.36, 6.27, 5.73]
               
                Mystery = [0.1, 0.2]
            
            
            
            
            if (default_bool == True and Rxx == 1 and sigma == 1):
               #######  D230831B_4_,          Rxx = 1      Conductivity  #########################
               
                GV = np.array([000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650])
           
                Const = [0.14, 0.14, 0.14, 0.12, 0.14, 0.14, 0.12, 0.12, 0.14, 0.14, 0.12, 0.16]
                B1 = [0.32, 0.50, 0.52, 0.75, 0.79, 0.81]
                B2 = [3.91, 3.85, 3.89, 3.87, 3.87, 3.95, 3.97]
                B3 = [4.06, 4.52, 4.82, 4.92, 4.98, 5.26, 5.34, 5.34, 5.38, 5.56, 5.58, 5.62]
               
            
            if (default_bool == True and Rxx == 2 and sigma == 1):
               #######  D230831B_4_,          Rxx = 2      Conductivity  #########################
               
                GV = np.array([000, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 650])
           
                Const = [0.14, 0.12, 0.13, 0.14, 0.18, 0.12, 0.12, 0.1, 0.12, 0.14, 0.14, 0.12]
                B1 = [0.32, 0.54, 0.58, 0.77, 0.8, 0.83]
                B2 = [3.85, 3.97, 4.07, 4.13, 3.95, 3.97, 3.95, 3.97, 3.97, 3.95, 3.93]
                B3 = [4.09, 4.54, 4.84, 4.94, 4.98, 5.26, 5.22, 5.34, 5.36, 5.52, 5.58, 5.62]
              
                Mystery = [0.1, 0.2]
            
        if NU == True:
        
            if (Rxx == 1 and sigma == 0):
               #######  D230831B_4_,          Rxx = 2      Conductivity  #########################
               
                GV = np.array([0, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
           
                Const = [0.24, 0.28, 0.28, 0.26, 0.26, 0.34, 0.30, 0.3, 0.3, 0.36, 0.3, 0.34, 0.38]
                B1 = [4.26, 4.92, 5.3, 5.58]
                B2 = [5.60, 5.65, 5.65, 5.67, 5.83, 5.87, 6.03, 5.71, 5.81, 5.87]
                #B3 = [4.09, 4.54, 4.84, 4.94, 4.98, 5.26, 5.22, 5.34, 5.36, 5.52, 5.58, 5.62]
              
                #Mystery = [0.1, 0.2]
        
        
        
        graph, ax1 = plt.subplots()  
        
        
        ax1.plot(GV, Const, c = 'r')
        m_Const, b_Const = np.polyfit(GV, Const, 1)
        
        
        ax1.plot(GV[:len(B1)], B1, c = 'orange', label = "$Main Peak$")
        m_B1, b_B1 = np.polyfit(GV[:len(B1)], B1, 1)
      
        
      
        ax1.plot(GV[-len(B2):], B2, c = 'orange')
        m_B2, b_B2 = np.polyfit(GV[-len(B2):], B2, 1)
      
        ax1.plot(GV[-len(B3):], B3, c = 'blue')
        m_B3, b_B3 = np.polyfit(GV[-len(B3):], B3, 1)
      
      
        #ax1.plot(GV, B3 , label = "$3^{rd}$ Peak")
        #m1_B3, b1_B3 = np.polyfit(GV[:-len(B1)+1], B3[:-len(B1)+1], 1)
        #m2_B3, b2_B3 = np.polyfit(GV[-len(B1):], B3[-len(B1):], 1)
        
        
       
        
        #if Rxx == 2:
        #    ax1.plot(GV[-len(Mystery):], Mystery, c = "purple", label = "Mystery")
        #    m_Mystery, b_Mystery = np.polyfit(GV[-len(Mystery):], Mystery, 1)
        #   
        #    #Save all calculated linear fit slopes to an array
        #    slopes = np.array([m_Const, m_B1, m_B2, m1_B3, m2_B3, m_Mystery]) * (2*c.e/c.h) * 1e-4 * 1000
        #else:
            
            
        #slopes = np.array([m_Const, m_B1, m_B2, m1_B3, m2_B3]) * (2*c.e/c.h) * 1e-4 * 1000
        slopes = np.array([m_Const, m_B1, m_B2, m_B3]) * (2*c.e/c.h) * 1e-4 * 1000
    
        
        if plot_fits:   
            
            ax1.plot(GV[:len(B1)], (GV[:len(B1)] * m_B1) + b_B1, "black", linestyle = '--')
            ax1.plot(GV[-len(B2):], (GV[-len(B2):] * m_B2) + b_B2, "black", linestyle = '--')
            ax1.plot(GV[-len(B3):], (GV[-len(B3):] * m_B3) + b_B3, "black", linestyle = '--')

            #ax1.plot(GV[:-len(B1)+1], (GV[:-len(B1)+1] * m1_B3) + b1_B3, c = "black", linestyle = '--')
            #ax1.plot(GV[-len(B1):], (GV[-len(B1):] * m2_B3) + b2_B3, c = "Black", linestyle = '--')
           
            
            ax1.annotate(text="{:.2e}$cm^-\u00b2/V$".format(round(slopes[1], -8)),
                     xy=[0.0,0.8], fontsize=12,
                     xycoords='axes fraction')
            ax1.annotate(text="{:.2e}$cm^-\u00b2/V$".format(round(slopes[2], -8)),
                     xy=[0.65,0.75], fontsize=12,
                     xycoords='axes fraction')
            ax1.annotate(text="{:.2e}$cm^-\u00b2/V$".format(round(slopes[3], -8)),
                     xy=[0.65,0.2], fontsize=12,
                     xycoords='axes fraction')
            #ax1.annotate(text="{:.2e}$cm^-\u00b2/V$".format(round(slopes[3], -8)),
            #         xy=[0.05,0.9],
            #         xycoords='axes fraction')
            #ax1.annotate(text="{:.2e}$cm^-\u00b2/V$".format(round(slopes[4], -8)),
            #         xy=[0.7,0.9],
            #         xycoords='axes fraction')
            #if Rxx == 2:
            #    ax1.plot(GV[-len(B1):], (GV[-len(B1):] * m_Mystery) + b_Mystery, c = "Black", linestyle = '--')
            #    ax1.annotate(text="{:.2e}$cm^-\u00b2/V$".format(round(slopes[5], -8)),
            #             xy=[0.6,0.7],
            #             xycoords='axes fraction')
            
            
        ax1.set_xlabel("Gave Voltage ($mV$)")
        ax1.set_ylabel("Magnetic Field ($T$)")
        ax1.set_ylim(0, 7)
        ax1.set_title("Peak locations in FFT results vs. Gate voltage")
        secax = ax1.secondary_yaxis('right', functions=(QFT.B_to_n, QFT.n_to_B))
        secax.set_ylabel("Carrier Conc. ($cm^{-2}$)")
        
            
        ax1.annotate(text=r"$B$ range of FFT = ["+ np.format_float_positional(B_range[0], unique = False, precision=2)+ r" T, "+np.format_float_positional(B_range[1], unique = False, precision=2)+r"T]",
                 xy=[0.05,0.2],
                 xycoords='axes fraction')
        #if lockin2xx_bool == True:
        #    if Rxx == 1:
        #        ax1.annotate(text="Rxx_1, D230831B_4_", xy=[0.05,0.15], xycoords='axes fraction')
        #    elif Rxx == 2:
        #        ax1.annotate(text="Rxx_2, D230831B_4_", xy=[0.05,0.15], xycoords='axes fraction')
        #else:
        #    if Rxx == 1:
        #        ax1.annotate(text="Rxx_1, D230831B_2_", xy=[0.05,0.15], xycoords='axes fraction')
        #    elif Rxx == 2:
        #        ax1.annotate(text="Rxx_2, D230831B_2_", xy=[0.05,0.15], xycoords='axes fraction')
        
        ax1.grid()
        #ax1.legend()
        
        
    
    
    '''
    ###Linear Fits
    m_const, b_const = np.polyfit(GV, Const, 1)
    m_B1, b_B1 = np.polyfit(GV[-len(B1):], B1, 1)
    m_B2, b_B2 = np.polyfit(GV[-len(B2):], B2, 1)
    m1_B3, b1_B3 = np.polyfit(GV[:-len(B1)+1], B3[:-len(B1)+1], 1)
    m2_B3, b2_B3 = np.polyfit(GV[-len(B1):], B3[-len(B1):], 1)
    #m_Mystery, b_myst = np.polyfit(GV[-len(Mystery):], Mystery, 1)
    
    #print([m_const, m_B1, m_B2, m_B3])
     
    
    ax1.plot(GV[:-len(B1)+1], (GV[:-len(B1)+1] * m1_B3) + b1_B3, c = "black", linestyle = '--')
    ax1.plot(GV[-len(B1):], (GV[-len(B1):] * m2_B3) + b2_B3, c = "Black", linestyle = '--')
    ax1.plot(GV[-len(B1):], (GV[-len(B1):] * m_B1) + b_B1, "black", linestyle = '--')
    
    
    
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