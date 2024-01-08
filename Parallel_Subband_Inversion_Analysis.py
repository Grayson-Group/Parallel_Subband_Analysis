import csv
import math
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
from bisect import bisect_left, bisect_right



'''
Keeping track of data

02 runs 0mV gate voltage:
    
    Lockin 1(Rxx) Phase:
        at B = 8T, Rotate = 10deg
        at B = 4T, Rotate = 7deg
        at B = 0.8T, Rotate = 5deg   (Note: rotate of 10deg gives Rxx offset of ~2ohms)
                
                
                
    Lockin 2(Rxy) Phase:
        at B = 8T, Rotate = 6.5deg
        at B = 4T, Rotate = 4.5deg
        at B = 0.5T, Rotate = 3.5deg  

        Current correction:
            at B = 8T, increase 2.61%
            at B = 4T, increase 2.25%
            at B = 0.5T, increase 1.8%
            
        Offset (using B = 8T, nu = 1 settings)
            B = 4T,  38 ohm,  (0.29% error)
            B = 0.5T, 10 ohms,  (0.62% error)
            
    Lockin 3(Rxx_2) Phase:
        at B = 8T, Rotate = 8deg
        at B = 4T, Rotate = 6deg
        at B = 0.5T, Rotate = 6deg
        
        
    Capacitance (Using current correction for B=8T and using phase rotation for the specified B/lockin values:)
        Lockin 1(Rxx):
            nu = 1: Vxx_y = -194 ohms,   wC = 2.911E-7
            nu = 2: Vxx_y = -99 ohms,    wC = 5.943E-7
            nu = 16: Vxx_y = -12.3 ohms,  wC = 4.726E-6
            
            
        Lockin 3(Rxx_2): 
            nu = 1: Vxx_y = 706 ohms,   wC = 1.06E-6
            nu = 2: Vxx_y = 231 ohms,   wC = 1.387E-6
            nu = 16: Vxx_y = 16 ohms,   wC = 6.15E-6



04 runs "Short SdH":
    

    Lockin 1(Rxx) Phase:
        at B = 8T, Rotate = 10.6deg
        at B = 4T, Rotate = 7.3deg
        at B = 0.5T, Rotate = 3deg   (Note: rotate of 10.6deg gives Rxx offset of ~0.5ohms)

    Lockin 2(Rxx_2) Phase:
        at B = 8T, Rotate = 10.6deg
        at B = 4T, Rotate = 8.6deg
        at B = 0.5T, Rotate = 9deg   (Note: rotate of 10.6deg gives Rxx offset of ~0ohms)

    Lockin 3(Rxy) Phase:
        at B = 8T, Rotate = 5.5deg
        at B = 4T, Rotate = 3.8deg
        at B = 0.5T, Rotate = 3deg   (Note: rotate of 10.6deg gives Rxx offset of ~0.5ohms)

            Current correction:
                at B = 8T, increase 2.39%
                at B = 4T, increase 2.03%
                at B = 0.5T, increase 1.52%
    
            Offset (using B = 8T, nu = 1 settings)
                B = 4T,  40 ohm,  (0.31% error)
                B = 0.5T, 12 ohms,  (0.74% error)
                
                
                
    Capacitance (Using current correction for B=8T and using phase rotation for the specified B/lockin values:)
    
        w = 13.8Hz
        
        Lockin 1(Rxx):
            nu = 1: Vxx_y = -107 ohms,   wC = 1.606E-7
            nu = 2: Vxx_y = -55 ohms,    wC = 3.302E-7
            nu = 16: Vxx_y = -7.3 ohms,  wC = 2.805E-6
            
            
        Lockin 2(Rxx_2): 
            nu = 1: Vxx_y = 395 ohms,   wC = 5.928E-7
            nu = 2: Vxx_y = 130 ohms,   wC = 7.805E-7
            nu = 16: Vxx_y = 10 ohms,   wC = 3.842E-6
    
    
    
    
    
    
    
    PHASE using RATIO OF R_XY PLATEAUS
    
        Lockin 2:   lockin2xx = False
                    11.5 Deg  (This agrees well for nu = 1, 2, 3, 4)
                    Current Adjust: Isclar = 0.9707     (all plateaus off by <2 Ohms)
        
        
        Lockin 3:   lockin2xx = True
                    12.1 Deg  (This agrees well for nu = 1, 2, 3, 4)
                    Current Adjust: Iscalar = 0.9701    (all plateaus off by < 4 Ohms)
'''


'''
###########################
USE THESE PARAMETERS:
  I = 2e-6
  Iscalar = 0.97
  Rotate = [10, 11.5, 12.1]   ([Lockin_1 phase, Lockin_2 phase, Lockin_3 phase])
    
###########################
'''




def ParallelAnalysis(lockin2XX: bool, gradient: bool, Rxx_1or2: int, Vg = 000,
                     I = 2e-6, Iscaler = 0.97, Rotate = [10,11.5,12.1], ne = 4E15, B_start = 0, B_end = -1):




    '''
        Vg: Gate voltage (mV) (selects file of this gate voltage)
        I: Current (Amps)
        Iscaler: Constant to multiply current by
        Rxx_1or2: (int, 1 or 2), chooses whether to use Rxx or Rxx_2 data when performing FFT analysis
        lockin2XX: (Boolean), if True the QFT.get_dat_data function will grab Rxx_2 data from lockin 2,
                    otherwise it will grab Rxx_2 data from lockin 3.
                    NOTE: True will automatically select files with file_name ending in (_3_ or _4_) as these files all use lockin 2 for Rxx
        geo_fact: (float), geometric factor that Rxx must be scaled by to become Rho_xx
        Rotate: (List of DEGREES with length 3), each element is a complex phase change [Lockin_1 Phase, Lockin_2 Phase, Lockin_3 Phase]
        ne: Carrier concentration in well, assumed roughly constant across B field, used for Rho parallel calculations
        B_start, B_end: (float), start and ending values (in Tesla) of B field to observe and analyse
        gradient: (Boolean), determines if gradient of Rxx or raw Rxx should be used when calculating FFT
    '''
    
    
    
    #######FIX _3 CONTACT PLOT
    #They are all differ Rxx configurations, need to label plots as such
    
    
    PlotRAWXX = 1
    PlotRAWXY = 1
    PlotINVXX = 0
    PlotINVXY = 0
    PlotFFTXX = 1
    

    SaveRAWXX = 0
    SaveRAWXY = 0
    SaveINVXX = 0
    SaveINVXY = 0
    SaveFFTXX = 0
    
    
    smoothing = 1        ###Option to smooth jaggady low B data before performing FFT
    RemoveFFTSpikes = 0  ###User defines regions of FFT to remove, then FFT is inverted back to resistance vs. 1/B data
    
    
    file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\Parallel_Subband_Analysis\D230831B 2nd cooldown\Full Sweeps"
    #file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\ETH Zurich Materials\\Code with Chris\\Parallel_Subband_Analysis\\D230831B 2nd cooldown\\Full Sweeps"
    #file_path = "D230831B 2nd cooldown/Full Sweeps"

    
    if lockin2XX == False:
        file_name = "D230831B_2_"
        file_name = file_name + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        geo_fact = (0.5/2.65)
    
    if lockin2XX == True:
        file_name = "D230831B_4_"            #Use for almost all full sweeps
        
        #file_name =  "D230831B_3_Contacts_"   #For Contact comparison _3 sweep
        #file_name = "D230831B_4_LowField_"      #For initial 0meV low B field run
        #file_name = "D230831B_4_Last_"       #For final 0meV low B field run
        #file_name = "D230831B_4_Negative_"   #For -200mV gate run
        
        geo_fact = (0.5/1.325)
        
        if type(Vg) == int:
            file_name = file_name + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
            
        
        if type(Vg) == str:    
            if Vg in ["D230831B_3_Contacts_", "D230831B_4_LowField_", "D230831B_4_Last_"]:
                file_name = Vg
                file_name = file_name + np.format_float_positional(000,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
                Vg = 000
            elif Vg == "D230831B_4_Negative_":
                file_name = Vg
                file_name = file_name + np.format_float_positional(200,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
                Vg = -200
            else:
                raise NameError('No usuable Vg_val detected')
            
        
        
    
    D230831B_5_data = QFT.get_dat_data(file_path, file_name, ["ETH"], lockin2XX, 
                                       has_header=True, data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y", "lockin3 x", "lockin3 y"],
                                       VoverI = (1/(I*Iscaler)))
    
    ### Here we filter by B field for values greater than B_start and less than B_end
    if B_end != -1:
        D230831B_5_data = D230831B_5_data[D230831B_5_data.An_Field > B_start]
        D230831B_5_data = D230831B_5_data[D230831B_5_data.An_Field < B_end]
    else:
        D230831B_5_data = D230831B_5_data[D230831B_5_data.An_Field > B_start]


    #Ignore first and last 50 data points
    #Rxx_x = D230831B_5_data.Rxx_x[50:-50]
    #Rxx_y = D230831B_5_data.Rxx_y[50:-50]
    #Rxy_x = D230831B_5_data.Rxy_x[50:-50]
    #Rxy_y = D230831B_5_data.Rxy_y[50:-50]
    #Rxx_x2 = D230831B_5_data.Rxx_x2[50:-50]
    #Rxx_y2 = D230831B_5_data.Rxx_y2[50:-50]
    #An_Field = D230831B_5_data.An_Field[50:-50]
    
    Rxx_x = D230831B_5_data.Rxx_x
    Rxx_y = D230831B_5_data.Rxx_y
    Rxy_x = D230831B_5_data.Rxy_x
    Rxy_y = D230831B_5_data.Rxy_y
    Rxx_x2 = D230831B_5_data.Rxx_x2
    Rxx_y2 = D230831B_5_data.Rxx_y2
    An_Field = D230831B_5_data.An_Field
    
    #Rxx_grad = np.empty(len(Rxx_x))
    #Rxx_grad = np.gradient(Rxx_x)
    #Rxy_grad = np.empty(len(Rxy_x))
    #Rxy_grad = np.gradient(Rxy_x)


    
    ###ROTATE DATA BY USER DEFINED PHASE####
    
    if lockin2XX == True:
        Rxx_x, Rxx_y = QFT.ComplexRotate(Rxx_x, Rxx_y, Rotate[0])
        Rxx_x2, Rxx_y2 = QFT.ComplexRotate(Rxx_x2, Rxx_y2, Rotate[1])  #Lockin 2 measures RXX
        Rxy_x, Rxy_y = QFT.ComplexRotate(Rxy_x, Rxy_y, Rotate[2])
    else:
        Rxx_x, Rxx_y = QFT.ComplexRotate(Rxx_x, Rxx_y, Rotate[0])
        Rxx_x2, Rxx_y2 = QFT.ComplexRotate(Rxx_x2, Rxx_y2, Rotate[2])  #Lockin 3 measures RXX
        Rxy_x, Rxy_y = QFT.ComplexRotate(Rxy_x, Rxy_y, Rotate[1])
    
    
    
    #CREATE MAIN DATAFRAME
    inv = 0
    inv = pd.DataFrame({'B_field': An_Field,
                        'Rxx': Rxx_x,
                        'Rxx_y':Rxx_y,
                        'Rxx2': Rxx_x2,
                        'Rxx2_y':Rxx_y2,
                        'Rxy': Rxy_x,
                        'Rxy_y':Rxy_y})
    inv.sort_values(by='B_field',inplace=True,ignore_index=True)
    
    
    
    ###################################################
    ###################################################
    #         Parallel Subband Analysis               #
    ###################################################
    ###################################################
    
    
    
    
    
    
    
    ### Convert Resistance to Resistivity
    rho_xx_tot = inv.Rxx*geo_fact  #R_xx multiplied by geometric factor
    rho_xy_tot = inv.Rxy           #rho_xy  is equal to Rxy
    rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
    
    
    
    
    #### ne*c.e/An_Field     OR      nu*c.e**2/c.h

    
    nu = 1
    rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 2
    rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 3
    rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 4
    rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 5
    rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 6
    rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 7
    rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    nu = 8
    rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
    
    
    #inv[['p_xx_tot', 'p_xy_tot', 'p_det_tot', 'rho_xx_par_nu1', 'rho_xy_par_nu1',
    #    'rho_xx_par_nu2', 'rho_xy_par_nu2', 'rho_xx_par_nu3', 'rho_xy_par_nu3',
    #    'rho_xx_par_nu4', 'rho_xy_par_nu4', 'rho_xx_par_nu5', 'rho_xy_par_nu5',
    #    'rho_xx_par_nu6', 'rho_xy_par_nu6', 'rho_xx_par_nu7', 'rho_xy_par_nu7',
    #    'rho_xx_par_nu8', 'rho_xy_par_nu8']]  =  [rho_xx_tot, rho_xy_tot, rho_det_tot, 
    #                            rho_xx_par_nu1, rho_xy_par_nu1, rho_xx_par_nu2, rho_xy_par_nu2,
    #                            rho_xx_par_nu3, rho_xy_par_nu3, rho_xx_par_nu4, rho_xy_par_nu4,
    #                            rho_xx_par_nu5, rho_xy_par_nu5, rho_xx_par_nu6, rho_xy_par_nu6,
    #                            rho_xx_par_nu7, rho_xy_par_nu7, rho_xx_par_nu8, rho_xy_par_nu8]
    inv['p_xx_tot'] = rho_xx_tot                        
    inv['p_xy_tot'] = rho_xy_tot
    inv['p_det_tot'] = rho_det_tot
    inv['rho_xx_par_nu1'] =  rho_xx_par_nu1
    inv['rho_xy_par_nu1'] = rho_xy_par_nu1
    inv['rho_xx_par_nu2'] = rho_xx_par_nu2
    inv['rho_xy_par_nu2'] = rho_xy_par_nu2
    inv['rho_xx_par_nu3'] = rho_xx_par_nu3
    inv['rho_xy_par_nu3'] = rho_xy_par_nu3
    inv['rho_xx_par_nu4'] = rho_xx_par_nu4
    inv['rho_xy_par_nu4'] = rho_xy_par_nu4
    inv['rho_xx_par_nu5'] = rho_xx_par_nu5
    inv['rho_xy_par_nu5'] = rho_xy_par_nu5
    inv['rho_xx_par_nu6'] = rho_xx_par_nu6
    inv['rho_xy_par_nu6'] = rho_xy_par_nu6         
    inv['rho_xx_par_nu7'] = rho_xx_par_nu7
    inv['rho_xy_par_nu7'] = rho_xy_par_nu7
    inv['rho_xx_par_nu8'] = rho_xx_par_nu8
    inv['rho_xy_par_nu8'] = rho_xy_par_nu8
    
    inv.sort_values(by='B_field',inplace=True,ignore_index=True)
   


    nu_bounds = []
    
    
    '''
    ###  Nu bounds for Rxy = Lock- in 2 (0mV)
    
    nu_bounds.append((0,0)) # nu = 0
    nu_bounds.append((1450,1550)) # nu = 1
    nu_bounds.append((960,1060)) # nu = 2
    nu_bounds.append((715,775)) # nu = 3
    nu_bounds.append((575,625)) # nu = 4
    nu_bounds.append((500,540)) # nu = 5
    nu_bounds.append((450,480)) # nu = 6
    nu_bounds.append((420,440)) # nu = 7
    nu_bounds.append((380,408)) # nu = 8

    ###  Nu bounds for Rxy = Lock- in 3 (0mV)
    
    nu_bounds.append((0,0)) # nu = 0
    nu_bounds.append((1550,1730)) # nu = 1
    nu_bounds.append((1075,1175)) # nu = 2
    nu_bounds.append((820,870)) # nu = 3
    nu_bounds.append((680,730)) # nu = 4
    nu_bounds.append((500,540)) # nu = 5
    nu_bounds.append((450,480)) # nu = 6
    nu_bounds.append((420,440)) # nu = 7
    nu_bounds.append((380,408)) # nu = 8
    '''

    nu_bounds.append((0,0)) # nu = 0
    nu_bounds.append((1550,1730)) # nu = 1
    nu_bounds.append((1075,1175)) # nu = 2
    nu_bounds.append((820,870)) # nu = 3
    nu_bounds.append((680,730)) # nu = 4
    nu_bounds.append((500,540)) # nu = 5
    nu_bounds.append((450,480)) # nu = 6
    nu_bounds.append((420,440)) # nu = 7
    nu_bounds.append((380,408)) # nu = 8


    #######  PLOTTING  ########

    plt.close('all')  #Start fresh

    

    if PlotRAWXX == True:
        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])
        plt.figure("RAWXX")
        plt.title("RAWXX, Vg = " + str(Vg) + "mV")
        #plt.plot(inv.Rxx)
        #plt.xlim([0.51, 0.53])
        #plt.ylim([0, 10])
        plt.plot(inv.B_field,inv.Rxx, c = 'b', label = "Rxx")
        #plt.plot(inv.B_field, inv.Rxx_y, c = 'b')
        #plt.plot(inv.B_field, inv.Rxx2, c = 'orange',label = "Rxx_2")
        #plt.plot(inv.B_field, inv.Rxx2_y, c = 'orange')
        #plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        #plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        #plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        #plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.ylabel("$R_{xx} (\Omega)$")
        plt.xlabel("B (T)")
        plt.grid()
        plt.legend()

        plt.figure("RAWXX_InvB")
        plt.plot(1/(inv.B_field),inv.Rxx, c = 'b', label = "Rxx")
        plt.title("RAWXX VS 1/B, Vg = " + str(Vg) + "mV")
        plt.ylabel("$R_{xx} (\Omega)$")
        plt.xlabel("1/B $(T^{-1})$")
        plt.grid()
        plt.legend()
        
        if SaveRAWXX == True:
            if lockin2XX == False:
                plt.annotate(text=r"$Rxx2$ from Lock-In 3",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/RAWXX_2_" + str(Vg) + ".png")
            else:
                plt.annotate(text=r"$Rxx2$ from Lock-In 2",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/RAWXX_4_" + str(Vg) + ".png")
            
    
    
    
    if PlotRAWXY == True:
        plt.figure("RAWXY")
        plt.title("RAWXY, Vg = " + str(Vg) + "mV")
        plt.plot(inv.B_field,inv.Rxy)
        plt.plot(inv.B_field,inv.Rxy_y)
        #plt.xlim([4.0, 4.1])
        #plt.ylim([12925, 12950])
        #plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxy[nu_bounds[1][0]],inv.Rxy[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        #plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxy[nu_bounds[2][0]],inv.Rxy[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        #plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxy[nu_bounds[3][0]],inv.Rxy[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        #plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxy[nu_bounds[4][0]],inv.Rxy[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.ylabel("$R_{xy} (\Omega)$")
        plt.xlabel("B (T)")
        plt.grid()
        plt.legend()
        
        if SaveRAWXY == True:
            if lockin2XX == False:
                plt.annotate(text=r"$Rxx2$ from Lock-In 3",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/RAWXY_2_" + str(Vg) + ".png")
            else:
                plt.annotate(text=r"$Rxx2$ from Lock-In 2",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/RAWXY_4_" + str(Vg) + ".png")


    #inv Rho_xx plots
    if PlotINVXX == True:
        plt.figure("INVXX")
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label="Raw")
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',label=str(nu))
        #nu=2
        #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label="Raw")
        #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r', label=str(nu))
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        # nu=4
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        # nu=5
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        # nu=6
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity XX, $\nu$ = 1-"+str(nu) + " Vg = " + str(Vg) + "mV")
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()
    
        if SaveINVXX == True:
            if lockin2XX == False:
                plt.annotate(text=r"$Rxx2$ from Lock-In 3",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/INVXX_2_" + str(Vg) + ".png")
            else:
                plt.annotate(text=r"$Rxx2$ from Lock-In 2",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/INVXX_4_" + str(Vg) + ".png")
    
    
    
    #inv Rho_xy plots
    if PlotINVXY == True:
        plt.figure("INVXY")
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label="Raw")
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',label=str(nu))
        #nu=2
        #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k', label = "Raw")
        #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',label=str(nu))
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        # nu=4
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        # nu=5
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        # nu=6
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity XY, $\nu$ = 1-"+ str(nu) + " Vg = " + str(Vg) + "mV")
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()
        
        if SaveINVXY == True:
            if lockin2XX == False:
                plt.annotate(text=r"$Rxx2$ from Lock-In 3",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/INVXY_2_" + str(Vg) + ".png")
            else:
                plt.annotate(text=r"$Rxx2$ from Lock-In 2",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/INVXY_4_" + str(Vg) + ".png")



    if PlotFFTXX == 1:
        #User defined Rxx_1or2 determines if Rxx_x or Rxx_x2 is used for FFT calculations
        if gradient == True:
            if Rxx_1or2 == 1:
                D230831B_5_data["Rxx_grad"] = np.gradient(D230831B_5_data.Rxx_x, D230831B_5_data.An_Field)
            if Rxx_1or2 == 2:
                D230831B_5_data["Rxx_grad"] = np.gradient(D230831B_5_data.Rxx_x2, D230831B_5_data.An_Field)


            window = [-1,0]  #range of Rxx data points to use, set = [-1, 0] to use entire range of data
            
            #Take desired range of data, run preliminary data adjustments
            D230831B_5_R_pos , D230831B_5_B_pos = QFT.apodize_data(D230831B_5_data,["xx_grad"], order=1, background_mode="points",extra_point_inds=200, start_point=window[0],
                                                            chop_point = window[1], invert=False, show_plot=True)


        if gradient == False:
            window = [-1,0]  #range of Rxx data points to use, set = [-1, 0] to use entire range of data
            
            if Rxx_1or2 == 1:
                D230831B_5_R_pos , D230831B_5_B_pos = QFT.apodize_data(D230831B_5_data,["xx"], order=1, background_mode="points",extra_point_inds=200, start_point=window[0],
                                                        chop_point = window[1], invert=False, show_plot=True)
            if Rxx_1or2 == 2:
                D230831B_5_R_pos , D230831B_5_B_pos = QFT.apodize_data(D230831B_5_data,["xx2"], order=1, background_mode="points",extra_point_inds=200, start_point=window[0],
                                                        chop_point = window[1], invert=False, show_plot=True)
            

        
        #### FFT processing to get rid of ugly low B data (sharp triangles instead of sinusoidal data)  #######
        
        if smoothing == True:
            spacing = np.round(D230831B_5_B_pos[1] - D230831B_5_B_pos[0], 3)
            num_spacing = int(np.round(D230831B_5_B_pos[0]/spacing))
            
            #Extrapolate D230831B_5_B_pos from D230831B_5_B_pos[0] to  0T
    
            new_B = np.linspace(0, D230831B_5_B_pos[0]-spacing, num_spacing)
            added_length = len(new_B)
            new_B = np.append(new_B, D230831B_5_B_pos)
    
            #Pad D230831B_5_R_pos from 0T to D230831B_5_B_pos[0] with zeros
            zeros = np.zeros(added_length)
            new_R = np.append(zeros, D230831B_5_R_pos)
            #np.zeros()
            
            plt.figure()
            plt.plot(new_B, new_R)
            plt.title("Added B to 0T")
            plt.xlabel("B (T)")
            plt.ylabel("Rxx")
            
            
            #Perform FFT, convert x_axis to carrier concentration
            power = 13
            n_points = 2**power  ###n_points should be = a POWER OF 2
            D230831B_5_trans = ft.rfft(new_R,n=n_points)  
            D230831B_5_f_array =  np.arange(len(D230831B_5_trans)) / n_points / np.abs(spacing) *2*c.e / c.h  #Factor of 2 if FFT is of spin degenerate region
            
            
            print("Ratio of padded zeros to data: " + str(n_points/len(new_B)))
            
            fft_start = 30
            fft_cutoff = -1
            
            plt.figure()
            
            plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_5_trans[fft_start:fft_cutoff]), c='b', label = "real")
            plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_5_trans[fft_start:fft_cutoff]), c='r', label = "imaginary")
            plt.legend(loc = "lower right")
            
            plt.annotate(text=r"$B$ range = ["+ np.format_float_positional(B_start, unique = False, precision=1)+ r" T, "+np.format_float_positional(B_end, unique = False, precision=1)+r"T]",
                         xy=[0.65,0.95],
                         xycoords='axes fraction')
            plt.annotate(text=r"$T$ = 20 mK",
                         xy=[0.7,0.9],
                         xycoords='axes fraction')
            plt.ylabel(r'FFT Amplitude')
            plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
            plt.title(r'FFT in 1/B of Processed $R_\mathrm{xx}$ (20 mK), sample D230831B_5, $V_\mathrm{g}$ = ' + np.format_float_positional(Vg,precision=4,trim='-') + ' mV')
            #plt.xlim(0,5e11)
            
            scaling = 10
            new_t = np.append(D230831B_5_trans, np.zeros(len(D230831B_5_trans)*scaling))
            print(new_t)
            inverted_trans = ft.irfft(new_t)
            print(len(inverted_trans))
            print(len(new_t))
            
            len(D230831B_5_trans)/(n_points/len(new_B))
    
            
            
            interp_B = np.linspace(0, new_B[-1], len(new_R)*(scaling + 1))
            #interp_R = np.interp(interp_B, new_B, new_R)
            plt.figure()
            plt.plot(interp_B, inverted_trans[:len(interp_B)])   #NOTE: amplitude is off by ~ factor of 10
            plt.title("Smoothed Inverse FFT Results")
            plt.xlabel("B (T)")
            plt.ylabel("Rxx (Ohms)")
            
            
            #Rename data, remove added padded data
            cutoff = np.where(interp_B < D230831B_5_B_pos[0])
            D230831B_5_B_pos = interp_B[cutoff[0][-1]:]
            D230831B_5_R_pos = inverted_trans[cutoff[0][-1]:len(interp_B)]

        #Interpolate inbetween data points, possibly apply scaling
        D230831B_5_R_inv , D230831B_5_B_inv = QFT.interpolate_data(D230831B_5_R_pos, D230831B_5_B_pos, pad_zeros= False, interp_ratio=10,
                                                                                        invert=False, scaling_order=1.5, scaling_mode="None")
        #If order > 0, apply some amount of Norton-Beer apodization
        D230831B_5_R_inv = QFT.apod_NB(D230831B_5_R_inv, D230831B_5_B_inv, order=3, show_plot=True, invert=False)
        
        
        ####PAD ZEROS TO 0T^-1 
        OneOver_B_inv = 1/D230831B_5_B_inv
        
        spacing = np.round(OneOver_B_inv[1] - OneOver_B_inv[0], 5)
        num_spacing = int(np.round(OneOver_B_inv[0]/spacing))
        new_B = np.linspace(0 + spacing, OneOver_B_inv[0]-spacing, num_spacing - 1)
        added_length = len(new_B)
        new_B = np.append(new_B, OneOver_B_inv)
        OneOver_B_inv = new_B
        
        zeros = np.zeros(added_length)
        D230831B_5_R_inv = np.append(zeros, D230831B_5_R_inv)
        
        D230831B_5_B_inv = 1/OneOver_B_inv
        

                #####Error checking Plots######
        plt.figure()
        plt.plot(1/D230831B_5_B_inv , D230831B_5_R_inv)
        plt.xlabel("1/B $(T^{-1})$")
        plt.ylabel("Gradient of Rxx")
        if gradient == True:
            plt.ylabel("Deriv. of Rxx")
            plt.title("Post-Processing Derivative of Rxx vs 1/B")
        else:
            plt.ylabel("Rxx")
            plt.title("Post-Processing Rxx vs 1/B")

        # D230831B_5_delt_B_inv_av = np.abs(1/D230831B_5_B_pos[0] - 1/D230831B_5_B_pos[-1])/(0.5*(len(D230831B_5_B_pos)-1))
        

        #Calculate average space between 1/B datapoints
        D230831B_5_delt_B_inv = 1/D230831B_5_B_inv[1:-1] - 1/D230831B_5_B_inv[0:-2]
        D230831B_5_delt_B_inv_av = np.mean(D230831B_5_delt_B_inv)
        
        
        #Perform FFT, convert x_axis to carrier concentration
        power = 17
        n_points = 2**power  ###n_points should be = a POWER OF 2
        D230831B_5_trans = ft.rfft(D230831B_5_R_inv,n=n_points)  
        D230831B_5_f_array =  np.arange(len(D230831B_5_trans)) / n_points / np.abs(D230831B_5_delt_B_inv_av) *2*c.e / c.h  #Factor of 2 if FFT is of spin degenerate region
        
        print("Interpolation Ratio: " + str(len(D230831B_5_B_inv)/len(D230831B_5_B_pos)))
        print("# of data points used for FFT: 2^" + str(power))
        print("Ratio of padded zeros to data: " + str(n_points/len(D230831B_5_B_inv)))

        
        '''
        if Rxx_1or2 == 1:
            plt.figure()
            plt.plot(D230831B_5_data.An_Field, D230831B_5_data.Rxx_grad, c = 'r', label = "Grad")
            plt.plot(D230831B_5_data.An_Field, D230831B_5_data.Rxx_x, c = 'b', label = "R_{xx_x}")
        if Rxx_1or2 == 2:
            plt.figure()
            plt.plot(D230831B_5_data.An_Field, D230831B_5_data.Rxx_grad, c = 'r', label = "Grad")
            plt.plot(D230831B_5_data.An_Field, D230831B_5_data.Rxx_x2, c = 'b', label = "R_{xx_x2}")
        plt.title("Raw Rxx and Derivative of Rxx VS B")
        plt.legend()
        '''    
        
        
        ####MAIN FFT PLOT######

        #Define section of FFT to plot
        fft_start = 30
        fft_cutoff = -1
        
        plt.figure()
        # peaks = sig.find_peaks(1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff]), 
        #                        height = 0.1*np.amax(1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff])))
        # print(peaks)
        # peak_density = D230831B_5_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831B_5_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831B_5_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        
        
        #plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff]))
        plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_5_trans[fft_start:fft_cutoff]), c='b', label = "real")
        plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_5_trans[fft_start:fft_cutoff]), c='r', label = "imaginary")
        plt.legend(loc = "lower right")
        
        # for peak in peaks[0]:
        #     plt.scatter(1e-4*D230831B_5_f_array[fft_start+peak],1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff])[peak])
        #     plt.annotate(np.format_float_scientific(1e-4*D230831B_5_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.05e-4*D230831B_5_f_array[fft_start+peak],0.9e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff])[peak]])
        plt.annotate(text=r"$B$ range = ["+ np.format_float_positional(B_start, unique = False, precision=1)+ r" T, "+np.format_float_positional(B_end, unique = False, precision=1)+r"T]",
                     xy=[0.65,0.95],
                     xycoords='axes fraction')
        plt.annotate(text=r"$T$ = 20 mK",
                     xy=[0.7,0.9],
                     xycoords='axes fraction')
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed $R_\mathrm{xx}$ (20 mK), sample D230831B_5, $V_\mathrm{g}$ = ' + np.format_float_positional(Vg,precision=4,trim='-') + ' mV')
        #plt.xlim(0,5e11)
        

        ###TESTING, remove spikes from FFT, then inverse FFT to figure out source of noise
        if RemoveFFTSpikes == 1:
            
            region = [[120, 165], [255,315], [390, 450], [550,600]]  #Define regions of data to remove


            #use new_t array to copy all FFT data, but replace defined regions with zeros
            new_t = np.ones(len(D230831B_5_trans))
            plt.figure()
            for reg in region[:]:
                new_t[reg[0]:reg[1]] -= 1
                plt.plot(1e-4*D230831B_5_f_array[reg[0]:reg[1]],1e-6*np.abs(D230831B_5_trans[reg[0]:reg[1]]), linestyle = '--', c = 'r')
            new_t = new_t * D230831B_5_trans
            plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.abs(new_t[fft_start:fft_cutoff]))
            plt.xlim([0,5e11])
            plt.title(r'FFT in 1/B of Processed $R_\mathrm{xx}$ (20 mK), sample D230831B_5, $V_\mathrm{g}$ = ' + np.format_float_positional(Vg,precision=4,trim='-') + ' mV')
            plt.annotate(text=r"$B$ range = ["+ np.format_float_positional(B_start, unique = False, precision=1)+ r" T, "+np.format_float_positional(B_end, unique = False, precision=1)+r"T]",
                     xy=[0.65,0.8],
                     xycoords='axes fraction')
            plt.ylabel(r'FFT Amplitude')
            plt.legend(["Removed Data"])
            plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")

            #NOW: Inverse rFFT new_t with x axis as D230831B_5_f_array[fft_start:fft_cutoff]
            #yeet = ft.irfft(D230831B_5_trans, int(len(D230831B_5_B_inv)))

            inverted_trans = ft.irfft(new_t)
            #inverted_trans = ft.irfft(D230831B_5_trans)

            plt.figure()
            plt.title("Inverted FFT with peaks removed")
            plt.xlabel("1/B $(T^{-1})$")
            plt.ylabel("Grad. of Rxx")
            plt.annotate(text=r"$B$ range = ["+ np.format_float_positional(B_start, unique = False, precision=1)+ r" T, "+np.format_float_positional(B_end, unique = False, precision=1)+r"T]",
                     xy=[0.65,0.95],
                     xycoords='axes fraction')
            #plt.plot(1/D230831B_5_B_inv, inverted_trans[:int(len(inverted_trans)/padding)])
            plt.plot(1/D230831B_5_B_inv, inverted_trans[:len(D230831B_5_R_inv)])


        if SaveFFTXX == True:
            if lockin2XX == False:
                plt.annotate(text=r"$Rxx2$ from Lock-In 3",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/FFTXX_2_" + str(Vg) + ".png")
            else:
                plt.annotate(text=r"$Rxx2$ from Lock-In 2",
                     xy=[0.05,0.9],
                     xycoords='axes fraction')
                plt.savefig("plots/FFTXX_4_" + str(Vg) + ".png")
        
        
        #CREATE FFT DATAFRAME
        FFT = 0
        FFT = pd.DataFrame({'Trans':    D230831B_5_trans,
                            'f_array':  D230831B_5_f_array})
        
        
        #Gradient Dataframe
        Rxx_grad = 0
        Rxx_grad = pd.DataFrame({'B_field': D230831B_5_B_inv,
                            'Rxx_grad': D230831B_5_R_inv})
        Rxx_grad.sort_values(by='B_field',inplace=True,ignore_index=True)
    
      
    if PlotFFTXX == 0:
        FFT = None                 #Need to return something
        Rxx_grad = None
        
        
    return inv, FFT, Rxx_grad, nu_bounds

def get_closests(df, col, val):
    '''
    Look through a specific column in a dataframe and find the index of the value closest
    to the user inputed value
    
    Parameters
    ----------
    df : Dataframe
        Dataframe you want to sparse
    col : String
        Column name
    val : List of float
        All values you want to be found in df

    Returns
    -------
    idx : List of index

    '''
    idx = []
    for i in val:
        idx.append(bisect_left(df[col].values, i))
        #higher_idx = bisect_right(df[col].values, i)
    return idx


def scaling(ref_df, df, scale, bounds, plot):
    '''

    Parameters
    ----------
    ref_df : pandas dataframe
        Unbiased (0mV) data, to be used as a reference to find where 
        plateaus/valleys in Rxy/Rxx have shifted to
    df : pandas dataframe
        Any dataframe you wish to observe. This dataframe will not be altered in anyway,
        the boundaries of ref_df will be scaled and trasfered to df accordingly
    scale : float
        Scaling factor that the ref_df B-field values will be multiplied by
    bounds : list of float
        Enter the reference boundaries you wish you be scaled and transfered
        to the df dataframe
    plot : Bool
        Plot scaled data, scaled boundaries, and untouched data

    Returns
    -------
    trans_bounds : TYPE
        DESCRIPTION.

    '''
    scaled = ref_df.copy()
    scaled.B_field *= scale
    scaled_bounds = [scaled.B_field[bounds[0]], scaled.B_field[bounds[1]]]
    
    
    trans_bounds = get_closests(df, "B_field", scaled_bounds)
    
    if plot == True:
        
        plt.figure("RAWXX")
        plt.title("RAWXX, Vg =  mV")
        plt.plot(ref_df.B_field, ref_df.Rxx, c = 'gray', linestyle = "--", alpha = 0.5, label = "Ref")
        plt.plot(scaled.B_field, scaled.Rxx, c = 'r', label = "Scaled")
        plt.plot(df.B_field, df.Rxx, c = 'b', label = "Compare")
        #plt.scatter([scaled_bounds[0], scaled_bounds[1]], [scaled.Rxx[bounds[0]], scaled.Rxx[bounds[1]]], c = 'r')
        plt.scatter([df.B_field[trans_bounds[0]], df.B_field[trans_bounds[1]]], [df.Rxx[bounds[0]], df.Rxx[bounds[1]]], c = 'r')
        plt.legend()
    
    
    
   
    return trans_bounds  


if __name__ == "__main__":
    
    
    
    # D230831B Second Cooldown 150 mV low field continuous nu inversion
    if 1==0:
        file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\D230831B 2nd cooldown\02 full sweeps"
        Vg = 000
        
        #file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        file_name = "D230831B_4_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        I = 1e-6
        Iscaler = 1.0239
        
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],
                                           has_header=True,data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y", "lockin3 x", "lockin3 y"],
                                           VoverI = (1/(I*Iscaler)), lockin2XX = True)


        #Ignore first and last 50 data points
        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        Rxx_x2 = D230831B_5_data.Rxx_x2[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        
        
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        carrierconc = [4E15] #Roughly constant Electron concentration at low B
        
        plt.close('all')  #Start fresh
        
        plt.figure("RAWXX")  #Create 4 main plots
        plt.figure("RAWXY")
        plt.figure("INVXX")
        plt.figure("INVXY")
        
        
        for ne in carrierconc:
            rho_xx_tot = D230831B_5_data.Rxx_x[50:-50]*(0.5/2.65)   ####Wtf is this factor?
            rho_xy_tot = D230831B_5_data.Rxy_x[50:-50]
            rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
            # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
            
             #### ne*c.e/An_Field     OR       nu*c.e**2/c.   #######

            nu = 1
            rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 2
            rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 3
            rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 4
            rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 5
            rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 6
            rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 7
            rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            nu = 8
            rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            inv = 0
            inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                                'Rxx': D230831B_5_data.Rxx_x,
                                'Rxx_y':D230831B_5_data.Rxx_y,
                                'Rxx2': D230831B_5_data.Rxx_x2,
                                'Rxx2_y':D230831B_5_data.Rxx_y2,
                                'Rxy': D230831B_5_data.Rxy_x,
                                'Rxy_y':D230831B_5_data.Rxy_y,
                                'p_xx_tot': rho_xx_tot,
                                'p_xy_tot': rho_xy_tot,
                                'p_det_tot': rho_det_tot,
                                'rho_xx_par_nu1': rho_xx_par_nu1,
                                'rho_xy_par_nu1':rho_xy_par_nu1,
                                'rho_xx_par_nu2': rho_xx_par_nu2,
                                'rho_xy_par_nu2': rho_xy_par_nu2,
                                'rho_xx_par_nu3': rho_xx_par_nu3,
                                'rho_xy_par_nu3':rho_xy_par_nu3,
                                'rho_xx_par_nu4': rho_xx_par_nu4,
                                'rho_xy_par_nu4':rho_xy_par_nu4,
                                'rho_xx_par_nu5': rho_xx_par_nu5,
                                'rho_xy_par_nu5':rho_xy_par_nu5,
                                'rho_xx_par_nu6': rho_xx_par_nu6,
                                'rho_xy_par_nu6':rho_xy_par_nu6,
                                'rho_xx_par_nu7': rho_xx_par_nu7,
                                'rho_xy_par_nu7':rho_xy_par_nu7,
                                'rho_xx_par_nu8': rho_xx_par_nu8,
                                'rho_xy_par_nu8':rho_xy_par_nu8
                                })
            inv.sort_values(by='B_field',inplace=True,ignore_index=True)
           
            
           
            #If desired, rotate data by complex phase
            inv.Rxx, inv.Rxx_y = QFT.ComplexRotate(inv.Rxx, inv.Rxx_y, 9.9)
            inv.Rxy, inv.Rxy_y = QFT.ComplexRotate(inv.Rxy, inv.Rxy_y, 6.5)
            inv.Rxx2, inv.Rxx2_y = QFT.ComplexRotate(inv.Rxx2, inv.Rxx2_y, 8)
            
            
            nu_bounds = []
            nu_bounds.append((0,0)) # nu = 0
            nu_bounds.append((1550,len(inv.Rxx)-1)) # nu = 1
            nu_bounds.append((980,1230)) # nu = 2
            ######nu_bounds.append((1040,1250)) # nu = 2
            # nu_bounds.append((705,765)) # nu = 3
            # nu_bounds.append((575,625)) # nu = 4
            # nu_bounds.append((500,540)) # nu = 5
            # nu_bounds.append((450,480)) # nu = 6
            # nu_bounds.append((420,440)) # nu = 7
            # nu_bounds.append((380,408)) # nu = 8
    
            # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])
            plt.figure("RAWXX")
            plt.title("RAWX, Vg = " + str(Vg) + "mV")
            #plt.plot(inv.Rxx[0:500])
            #plt.plot(inv.Rxx)
            #plt.xlim([0.51, 0.53])
            #plt.ylim([0, 10])
            plt.plot(inv.B_field,inv.Rxx, c = 'b', label = "Lockin 1")
            #plt.plot(inv.B_field, inv.Rxx_y, c = 'b')
            plt.plot(inv.B_field, inv.Rxx2, c = 'orange',label = "Lockin 3")
            #plt.plot(inv.B_field, inv.Rxx2_y, c = 'orange')
            #plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
            #plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
            # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
            # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
            # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
            # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
            # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
            # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
            

            plt.ylabel("$R_{xx} (\Omega)$")
            plt.xlabel("B (T)")
            plt.grid()
            # plt.xlim(0,4)
            plt.legend()
            
            
            plt.figure("RAWXY")
            plt.title("RAWXY, Vg = " + str(Vg) + "mV")
            # plt.plot(inv.Rxx[0:500])
            # plt.plot(inv.Rxx)
            plt.plot(inv.B_field,inv.Rxy)
            plt.plot(inv.B_field,inv.Rxy_y)
            #plt.xlim([4.0, 4.1])
            #plt.ylim([12925, 12950])
            #plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxy[nu_bounds[1][0]],inv.Rxy[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
            #plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxy[nu_bounds[2][0]],inv.Rxy[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
            # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
            # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
            # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
            # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
            # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
            # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
            plt.ylabel("$R_{xy} (\Omega)$")
            plt.xlabel("B (T)")
            plt.grid()
            # plt.xlim(0,4)
            plt.legend()
    
    
            #inv Rho_xx plots
            plt.figure("INVXX")
            nu=1
            #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label="Raw")
            #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',label=str(nu))
            nu=2
            #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label="Raw")
            plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r', label=str(nu))
            # nu=3
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
            # nu=4
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
            # nu=5
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
            # nu=6
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
            # nu=7
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
            # nu=8
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
            plt.title(r"Parallel Resistivity XX, $\nu$ = 1-"+str(nu) + " Vg = " + str(Vg) + "mV")
            plt.ylabel(r"$\rho$ ($\Omega$)")
            plt.xlabel(r"$B$ (T)")
            plt.legend()
            
            
            
            
            
            #inv Rho_xy plots
            
            plt.figure("INVXY")
            nu=1
            #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label="Raw")
            #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',label=str(nu))
            nu=2
            #plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xy_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k', label = "Raw")
            plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xy_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',label=str(nu))
            # nu=3
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
            # nu=4
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
            # nu=5
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
            # nu=6
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
            # nu=7
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
            # nu=8
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
            # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
            plt.title(r"Parallel Resistivity XY, $\nu$ = 1-"+ str(nu) + " Vg = " + str(Vg) + "mV")
            plt.ylabel(r"$\rho$ ($\Omega$)")
            plt.xlabel(r"$B$ (T)")
            plt.legend()
            
            plt.figure("RAWXX")
            


    # D230831B Second Cooldown 500 mV data
    if 1==0:
        file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\D230831B 2nd cooldown\02 full sweeps"
        Vg = 500#[-200, -100, 0]
        file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True,data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxy_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 3
        rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 4
        rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 5
        rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 6
        rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 7
        rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 8
        rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2,
                            'rho_xx_par_nu3': rho_xx_par_nu3,
                            'rho_xy_par_nu3':rho_xy_par_nu3,
                            'rho_xx_par_nu4': rho_xx_par_nu4,
                            'rho_xy_par_nu4':rho_xy_par_nu4,
                            'rho_xx_par_nu5': rho_xx_par_nu5,
                            'rho_xy_par_nu5':rho_xy_par_nu5,
                            'rho_xx_par_nu6': rho_xx_par_nu6,
                            'rho_xy_par_nu6':rho_xy_par_nu6,
                            'rho_xx_par_nu7': rho_xx_par_nu7,
                            'rho_xy_par_nu7':rho_xy_par_nu7,
                            'rho_xx_par_nu8': rho_xx_par_nu8,
                            'rho_xy_par_nu8':rho_xy_par_nu8
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((1510,len(inv.Rxx)-1)) # nu = 1
        #####nu_bounds.append((940,1080)) # nu = 2
        nu_bounds.append((1040,1250)) # nu = 2
        # nu_bounds.append((705,765)) # nu = 3
        # nu_bounds.append((575,625)) # nu = 4
        # nu_bounds.append((500,540)) # nu = 5
        # nu_bounds.append((450,480)) # nu = 6
        # nu_bounds.append((420,440)) # nu = 7
        # nu_bounds.append((380,408)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        # plt.xlim(0,4)
        plt.legend()

        plt.figure()
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$\rho_\mathrm{xx}^\mathrm{tot}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',ls=":",label=r'$\rho_\mathrm{xx}^\mathrm{\|\|}$')
        nu=2
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',ls=":")
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        # nu=4
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        # nu=5
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        # nu=6
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()







    # D230831B Second Cooldown 450 mV data
    if 1==0:
        file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\D230831B 2nd cooldown\02 full sweeps"
        Vg = 450#[-200, -100, 0]
        file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True,data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 3
        rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 4
        rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 5
        rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 6
        rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 7
        rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 8
        rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2,
                            'rho_xx_par_nu3': rho_xx_par_nu3,
                            'rho_xy_par_nu3':rho_xy_par_nu3,
                            'rho_xx_par_nu4': rho_xx_par_nu4,
                            'rho_xy_par_nu4':rho_xy_par_nu4,
                            'rho_xx_par_nu5': rho_xx_par_nu5,
                            'rho_xy_par_nu5':rho_xy_par_nu5,
                            'rho_xx_par_nu6': rho_xx_par_nu6,
                            'rho_xy_par_nu6':rho_xy_par_nu6,
                            'rho_xx_par_nu7': rho_xx_par_nu7,
                            'rho_xy_par_nu7':rho_xy_par_nu7,
                            'rho_xx_par_nu8': rho_xx_par_nu8,
                            'rho_xy_par_nu8':rho_xy_par_nu8
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((1510,len(inv.Rxx)-1)) # nu = 1
        #####nu_bounds.append((940,1080)) # nu = 2
        nu_bounds.append((1040,1250)) # nu = 2
        # nu_bounds.append((705,765)) # nu = 3
        # nu_bounds.append((575,625)) # nu = 4
        # nu_bounds.append((500,540)) # nu = 5
        # nu_bounds.append((450,480)) # nu = 6
        # nu_bounds.append((420,440)) # nu = 7
        # nu_bounds.append((380,408)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        # plt.xlim(0,4)
        plt.legend()

        plt.figure()
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$\rho_\mathrm{xx}^\mathrm{tot}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',ls=":",label=r'$\rho_\mathrm{xx}^\mathrm{\|\|}$')
        nu=2
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',ls=":")
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        # nu=4
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        # nu=5
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        # nu=6
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()






    # D230831B Second Cooldown 220 mV data
    if 1==0:
        file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\D230831B 2nd cooldown\02 full sweeps"
        Vg = 220#[-200, -100, 0]
        file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True,data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 3
        rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 4
        rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 5
        rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 6
        rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 7
        rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 8
        rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2,
                            'rho_xx_par_nu3': rho_xx_par_nu3,
                            'rho_xy_par_nu3':rho_xy_par_nu3,
                            'rho_xx_par_nu4': rho_xx_par_nu4,
                            'rho_xy_par_nu4':rho_xy_par_nu4,
                            'rho_xx_par_nu5': rho_xx_par_nu5,
                            'rho_xy_par_nu5':rho_xy_par_nu5,
                            'rho_xx_par_nu6': rho_xx_par_nu6,
                            'rho_xy_par_nu6':rho_xy_par_nu6,
                            'rho_xx_par_nu7': rho_xx_par_nu7,
                            'rho_xy_par_nu7':rho_xy_par_nu7,
                            'rho_xx_par_nu8': rho_xx_par_nu8,
                            'rho_xy_par_nu8':rho_xy_par_nu8
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((1510,len(inv.Rxx)-1)) # nu = 1
        #####nu_bounds.append((940,1080)) # nu = 2
        nu_bounds.append((1040,1250)) # nu = 2
        # nu_bounds.append((705,765)) # nu = 3
        # nu_bounds.append((575,625)) # nu = 4
        # nu_bounds.append((500,540)) # nu = 5
        # nu_bounds.append((450,480)) # nu = 6
        # nu_bounds.append((420,440)) # nu = 7
        # nu_bounds.append((380,408)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        # plt.xlim(0,4)
        plt.legend()

        plt.figure()
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$\rho_\mathrm{xx}^\mathrm{tot}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',ls=":",label=r'$\rho_\mathrm{xx}^\mathrm{\|\|}$')
        nu=2
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',ls=":")
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        # nu=4
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        # nu=5
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        # nu=6
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()

    
    
    
    
    # D230831B Second Cooldown 200 mV data
    if 1==0:
        file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\D230831B 2nd cooldown\02 full sweeps"
        Vg = 200#[-200, -100, 0]
        file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True,data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 3
        rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 4
        rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 5
        rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 6
        rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 7
        rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 8
        rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2,
                            'rho_xx_par_nu3': rho_xx_par_nu3,
                            'rho_xy_par_nu3':rho_xy_par_nu3,
                            'rho_xx_par_nu4': rho_xx_par_nu4,
                            'rho_xy_par_nu4':rho_xy_par_nu4,
                            'rho_xx_par_nu5': rho_xx_par_nu5,
                            'rho_xy_par_nu5':rho_xy_par_nu5,
                            'rho_xx_par_nu6': rho_xx_par_nu6,
                            'rho_xy_par_nu6':rho_xy_par_nu6,
                            'rho_xx_par_nu7': rho_xx_par_nu7,
                            'rho_xy_par_nu7':rho_xy_par_nu7,
                            'rho_xx_par_nu8': rho_xx_par_nu8,
                            'rho_xy_par_nu8':rho_xy_par_nu8
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((1510,len(inv.Rxx)-1)) # nu = 1
        #####nu_bounds.append((940,1080)) # nu = 2
        nu_bounds.append((1040,1250)) # nu = 2
        # nu_bounds.append((705,765)) # nu = 3
        # nu_bounds.append((575,625)) # nu = 4
        # nu_bounds.append((500,540)) # nu = 5
        # nu_bounds.append((450,480)) # nu = 6
        # nu_bounds.append((420,440)) # nu = 7
        # nu_bounds.append((380,408)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        # plt.xlim(0,4)
        plt.legend()

        plt.figure()
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$\rho_\mathrm{xx}^\mathrm{tot}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',ls=":",label=r'$\rho_\mathrm{xx}^\mathrm{\|\|}$')
        # nu=2
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',ls=":")
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        # nu=4
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        # nu=5
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        # nu=6
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()
    
    
    
    
    
    
    
    # D230831B Second Cooldown 000 mV data
    if 1==0:
        file_path = r"C:\Users\Madma\Documents\Northwestern\Research (Grayson)\GaAs Degen Calc\Gate tests\D230831B 2nd cooldown\02 full sweeps"
        Vg = 000#[-200, -100, 0]
        file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True,data_headings=["variable x","lockin1 x", "lockin1 y", "lockin2 x", "lockin2 y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 3
        rho_xx_par_nu3 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu3 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 4
        rho_xx_par_nu4 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu4 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 5
        rho_xx_par_nu5 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu5 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 6
        rho_xx_par_nu6 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu6 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 7
        rho_xx_par_nu7 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu7 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 8
        rho_xx_par_nu8 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu8 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2,
                            'rho_xx_par_nu3': rho_xx_par_nu3,
                            'rho_xy_par_nu3':rho_xy_par_nu3,
                            'rho_xx_par_nu4': rho_xx_par_nu4,
                            'rho_xy_par_nu4':rho_xy_par_nu4,
                            'rho_xx_par_nu5': rho_xx_par_nu5,
                            'rho_xy_par_nu5':rho_xy_par_nu5,
                            'rho_xx_par_nu6': rho_xx_par_nu6,
                            'rho_xy_par_nu6':rho_xy_par_nu6,
                            'rho_xx_par_nu7': rho_xx_par_nu7,
                            'rho_xy_par_nu7':rho_xy_par_nu7,
                            'rho_xx_par_nu8': rho_xx_par_nu8,
                            'rho_xy_par_nu8':rho_xy_par_nu8
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((1410,1600)) # nu = 1
        nu_bounds.append((940,1080)) # nu = 2
        nu_bounds.append((705,765)) # nu = 3
        nu_bounds.append((575,625)) # nu = 4
        nu_bounds.append((500,540)) # nu = 5
        nu_bounds.append((450,480)) # nu = 6
        nu_bounds.append((420,440)) # nu = 7
        nu_bounds.append((380,408)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="purple",label=r"$\nu$= 5")
        plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="purple",label=r"$\nu$= 6")
        plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="purple",label=r"$\nu$= 7")
        plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="purple",label=r"$\nu$= 8")
        plt.grid()
        plt.legend()

        plt.figure()
        nu_colors = ['k','b','r','g']
        nu=1
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$\rho_\mathrm{xx}^\mathrm{tot}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color='b',ls=":",label=r'$\rho_\mathrm{xx}^\mathrm{\|\|}$')
        nu=2
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color='r',ls=":")
        nu=3
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu3[nu_bounds[nu][0]:nu_bounds[nu][1]],color='orange',ls=":")
        nu=4
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu4[nu_bounds[nu][0]:nu_bounds[nu][1]],color='purple',ls=":")
        nu=5
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu5[nu_bounds[nu][0]:nu_bounds[nu][1]],color='yellow',ls=":")
        nu=6
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu6[nu_bounds[nu][0]:nu_bounds[nu][1]],color='pink',ls=":")
        nu=7
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        nu=8
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()

        # plt.figure()
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.Rxx[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$R_\mathrm{xx}$')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color=nu_colors[nu],label=r'$\nu$ = '+str(nu))
        # plt.legend()




    
    if 1==0:
        file_path = "z:\\User\\Thomas\\036CC_D230831B\\01 first sweeps Rxx only"
        file = "000mV_0T_12T_Rxx.dat"
        
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file,R_ind = ["ETH"],has_header=True,
                                           data_headings=["variable x", "lockin1 x", "lockin1 y", "lockin1 x", "lockin1 y"])
        plt.figure() 
        # plt.xlim(4,6)
        plt.plot(D230831B_5_data.An_Field,1e6*D230831B_5_data.Rxx_x)
        plt.ylim(bottom=0,top=10000)
        plt.ylabel(r"$R_\mathrm{xx}$ ($\mathrm{\mu \Omega}$)")
        plt.xlabel(r"$B$ (T)")
        # plt.legend()
        plt.title(r'SdH Curve for D230831B_5, $V_\mathrm{g}$ = 0V, $T$ = 15 mK')


        # file = "19102023_154452.dat"
        
        # # print(file_names[i])
        # D230831B_5_data = QFT.get_dat_data(file_path,file,R_ind = ["ETH"],has_header=True,
        #                                    data_headings=["variable x", "lockin1 x", "lockin1 y", "lockin1 x", "lockin1 y"])
        # plt.figure() 
        # plt.plot(D230831B_5_data.An_Field,-1e3*D230831B_5_data.Rxy_x)
        # # plt.xlim(0,1)
        # # plt.ylim(bottom=0)
        # plt.ylabel(r"$R_\mathrm{xy}$ (m$\mathrm{\Omega}$)")
        # plt.xlabel(r"$B$ (T)")
        # # plt.legend()
        # plt.title(r'Hall Curve for D230831B_5, $V_\mathrm{g}$ = 0V, $T$ = 15 mK')
        
        # plt.figure()
        # for Vg in Vg_vals:
        #     file = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        #     # print(file_names[i])
        #     D230831B_5_data = QFT.get_dat_data(file_path,file,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
        #     scaling_1 = 9e-4
        #     knee = 400
        #     scaling_2 = 7e-4
        #     if Vg <=knee:
        #         new_B = D230831B_5_data.An_Field / (scaling_1*Vg + 1)
        #     else:
        #         new_B = D230831B_5_data.An_Field / (scaling_2*(Vg - knee) + scaling_1*knee + 1)
            
        #     plt.plot(new_B,D230831B_5_data.Rxy_x,label=str(Vg))
        #     # plt.xlim(0,3)
        # plt.ylim(bottom=0)
        # plt.ylabel(r"$R_\mathrm{xy}$")
        # plt.xlabel(r"$B_\mathrm{scaled}$ (T/V)")
        # plt.title(r'B-Scaled Hall Curves for D230831B_5 at various $V_\mathrm{g}$')
        # plt.legend()








    if 1==0:
        file_path = "Z:\\samples\\D230831B_5"
        # Vg_vals = [-200, -100, 0, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750,800]
        Vg_vals =[100,200,300,350, 375, 400, 450, 475, 500, 525, 550, 575]# [0, 100, 150, 200, 250, 300, 350, 400,  450,  500, 550, 600,]
        file_names = []
        plt.figure()
        for Vg in Vg_vals:
            file = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
            # print(file_names[i])
            D230831B_5_data = QFT.get_dat_data(file_path,file,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
            scaling_1 = 3e-4
            knee = 900
            scaling_2 = 7e-4
            if Vg <=knee:
                new_B = D230831B_5_data.An_Field / (scaling_1*Vg + 1)
            else:
                new_B = D230831B_5_data.An_Field / (scaling_2*(Vg - knee) + scaling_1*knee + 1)
            
            plt.plot(new_B,D230831B_5_data.Rxx_x,label=str(Vg))
            # plt.xlim(0,3)
        plt.ylim(bottom=0)
        plt.ylabel(r"$R_\mathrm{xx}$")
        plt.xlabel(r"$B_\mathrm{scaled}$ (T/V)")
        plt.legend()
        plt.title(r'B-Scaled SdH Curves for D230831B_5 at various $V_\mathrm{g}$')
        
        # plt.figure()
        # for Vg in Vg_vals:
        #     file = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        #     # print(file_names[i])
        #     D230831B_5_data = QFT.get_dat_data(file_path,file,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
        #     scaling_1 = 9e-4
        #     knee = 400
        #     scaling_2 = 7e-4
        #     if Vg <=knee:
        #         new_B = D230831B_5_data.An_Field / (scaling_1*Vg + 1)
        #     else:
        #         new_B = D230831B_5_data.An_Field / (scaling_2*(Vg - knee) + scaling_1*knee + 1)
            
        #     plt.plot(new_B,D230831B_5_data.Rxy_x,label=str(Vg))
        #     # plt.xlim(0,3)
        # plt.ylim(bottom=0)
        # plt.ylabel(r"$R_\mathrm{xy}$")
        # plt.xlabel(r"$B_\mathrm{scaled}$ (T/V)")
        # plt.title(r'B-Scaled Hall Curves for D230831B_5 at various $V_\mathrm{g}$')
        # plt.legend()






            




### D230831B_5 Kanne Data, Vg = 125 mV
    if 1==0:
        file_path = "Z:\\samples\\D230831B_5"
        Vg = 150#[-200, -100, 0]
        file_name = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h) * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((0,0)) # nu = 1
        nu_bounds.append((3640,3700)) # nu = 2
        nu_bounds.append((1885,1895)) # nu = 3

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="b",label=r"$\nu$ = 2")
        plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="r",label=r"$\nu$ = 3")
        plt.grid()
        plt.legend()

        # plt.figure()
        # nu_colors = ['k','b','r','g']
        # nu=2
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.Rxx[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$R_\mathrm{xx}$')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color=nu_colors[nu],label=r'$\nu$ = '+str(nu))
        # plt.legend()

        # plt.figure()
        # nu=3
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.Rxx[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$R_\mathrm{xx}$')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color=nu_colors[nu],label=r'$\nu$ = '+str(nu))
        # plt.legend()






### D230831B_5 Kanne Data, Vg = -100 mV
    if 1==0:
        file_path = "Z:\\samples\\D230831B_5"
        Vg = -100#[-200, -100, 0]
        file_name = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
        # print(file_names[i])
        D230831B_5_data = QFT.get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])

        Rxx_x = D230831B_5_data.Rxx_x[50:-50]
        Rxy_x = D230831B_5_data.Rxy_x[50:-50]
        An_Field = D230831B_5_data.An_Field[50:-50]
        Rxx_grad = np.empty(len(Rxx_x))
        Rxx_grad = np.gradient(Rxx_x)
        Rxy_grad = np.empty(len(Rxy_x))
        Rxy_grad = np.gradient(Rxy_x)


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        # names = [('rho_xx_par_nu1','rho_xy_par_nu1')]
        nu = 1
        rho_xx_par_nu1 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu1 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        nu = 2
        rho_xx_par_nu2 = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par_nu2 = (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)* rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                            'Rxx': D230831B_5_data.Rxx_x,
                            'Rxy': D230831B_5_data.Rxy_x,
                            'p_xx_tot': rho_xx_tot,
                            'p_xy_tot': rho_xx_tot,
                            'p_det_tot': rho_det_tot,
                            'rho_xx_par_nu1': rho_xx_par_nu1,
                            'rho_xy_par_nu1':rho_xy_par_nu1,
                            'rho_xx_par_nu2': rho_xx_par_nu2,
                            'rho_xy_par_nu2': rho_xy_par_nu2
                            })
        inv.sort_values(by='B_field',inplace=True,ignore_index=True)
        nu_bounds = []
        nu_bounds.append((0,0)) # nu = 0
        nu_bounds.append((0,0)) # nu = 1
        nu_bounds.append((3640,3700)) # nu = 2
        nu_bounds.append((1885,1895)) # nu = 3

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        # plt.figure()
        # plt.plot(inv.B_field,inv.Rxx)
        # plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="b",label=r"$\nu$ = 2")
        # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="r",label=r"$\nu$ = 3")
        # plt.grid()
        # plt.legend()

        plt.figure()
        nu_colors = ['k','b','r','g']
        nu=2
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.Rxx[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$R_\mathrm{xx}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu1[nu_bounds[nu][0]:nu_bounds[nu][1]],color=nu_colors[nu],label=r'$\nu$ = '+str(nu))
        plt.legend()

        plt.figure()
        nu=3
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.Rxx[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k',label=r'$R_\mathrm{xx}$')
        plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu2[nu_bounds[nu][0]:nu_bounds[nu][1]],color=nu_colors[nu],label=r'$\nu$ = '+str(nu))
        plt.legend()

    #ParallelAnalysis(Vg = 100, lockin2XX = True, I = 1e-6, Iscaler = 1.0, Rotate = [0,0,0], ne = 4E15)
        
        # plt.figure()
        # plt.plot(An_Field,Rxx_x/np.amax(Rxx_x),label=r"Rxx")
        # plt.plot(An_Field,Rxx_grad/np.amax(Rxx_grad),label=r"dRxx")
        # plt.plot(An_Field,Rxx_x/np.amax(Rxx_x) - Rxx_grad/np.amax(Rxx_grad),label=r"Rxx - dRxx")
        # plt.plot(An_Field,Rxy_x/np.amax(Rxy_x),label=r"Rxy")
        # print(inv.B_field[nu_bounds[1][0]])
        # plt.plot(An_Field,Rxy_grad/np.amax(Rxy_grad),label=r"dRxy")
        # plt.plot(An_Field,Rxy_x/np.amax(Rxy_x) - Rxy_grad/np.amax(Rxy_grad),label=r"Rxy - dRxy")
        # plt.title(r"R_{xx} gradient")
        # plt.grid()
        # plt.legend()









#plt.show()