### QHE Data plotting sample D181211Ai, date 8/3/23
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


#Current should be ~2uA
#2nA with 1000x voltage gain




def get_csv_data(file_path : str,file_name: str,R_ind : list, VoverI = 1/(1e-6)) :
    os.chdir(file_path)
    file = file_name
    data = pd.read_csv(file)
    if "xy"in R_ind:
        data['Rxy_x'] = data.Vxy_x * VoverI # R = V/I, I = 1 uA
        data['Rxy_y'] = data.Vxy_y * VoverI
    elif "xx"in R_ind:
        data['Rxx_x'] = data.Vxx_x * VoverI
        data['Rxx_y'] = data.Vxx_y * VoverI 
    elif "yy"in R_ind:
        data['Ryy_x'] = data.Vyy_x * VoverI
        data['Ryy_y'] = data.Vyy_y * VoverI 
    elif "yx"in R_ind:
        data['Ryx_x'] = data.Vyx_x * VoverI
        data['Ryx_y'] = data.Vyx_y * VoverI

    return data


def get_dat_data(file_path : str, file_name: str, R_ind : list, lockin2XX: bool, 
                 VoverI = 1/(1e-6), has_header=False, data_headings =[]):
    '''data headings_format: [An_Field, Rxx_x, Rxx_y, Rxy_x, Rxy_y]'''
    os.chdir(file_path)
    # data = pd.read_fwf(file_name,infer_nrows=5,delim_whitespace=True)
    if data_headings == []:
        data = pd.read_table(file_name,delim_whitespace=True)
        if "xy"in R_ind:
            data['Rxy_x'] = data.Vxy_x * VoverI # R = V/I, I = 1 uA
            data['Rxy_y'] = data.Vxy_y * VoverI
        elif "xx"in R_ind:
            data['Rxx_x'] = data.Vxx_x * VoverI
            data['Rxx_y'] = data.Vxx_y * VoverI 
        elif "yy"in R_ind:
            data['Ryy_x'] = data.Vyy_x * VoverI
            data['Ryy_y'] = data.Vyy_y * VoverI 
        elif "yx"in R_ind:
            data['Ryx_x'] = data.Vyx_x * VoverI
            data['Ryx_y'] = data.Vyx_y * VoverI
        elif "ETH" in R_ind:
            pass
    else:
        if has_header == False:
            data = pd.read_table(file_name,delimiter="\t",header=None,names=data_headings)
            # print(data)
            # data headings_format: [An_Field, Rxx_x, Rxx_y, Rxy_x, Rxy_y]

        elif has_header == True:
            if lockin2XX == False:
                data = pd.read_table(file_name,header=0)
                data['An_Field'] = data[data_headings[0]]  # data_headings 0 = An_field
                data['Rxx_x'] = data[data_headings[1]] * VoverI    # data_headings 1 = Rxx_x
                data['Rxx_y'] = data[data_headings[2]] * VoverI    # data_headings 2 = Rxx_y
                data['Rxy_x'] = data[data_headings[3]] * VoverI * -1    # data_headings 3 = Rxy_x
                data['Rxy_y'] = data[data_headings[4]] * VoverI * -1   # data_headings 4 = Rxy_y
                
                if len(data_headings) > 5:
                    data['Rxx_x2'] = data[data_headings[5]] * VoverI    # data_headings 5 = Rxx_x
                    data['Rxx_y2'] = data[data_headings[6]] * VoverI    # data_headings 6 = Rxx_y
           
            if lockin2XX == True:   
                data = pd.read_table(file_name,header=0)
                data['An_Field'] = data[data_headings[0]]  # data_headings 0 = An_field
                data['Rxx_x'] = data[data_headings[1]] * VoverI    # data_headings 1 = Rxx_x
                data['Rxx_y'] = data[data_headings[2]] * VoverI    # data_headings 2 = Rxx_y
                data['Rxy_x'] = data[data_headings[5]] * VoverI    # data_headings 5 = Rxy_x
                data['Rxy_y'] = data[data_headings[6]] * VoverI    # data_headings 6 = Rxy_y
                
                if len(data_headings) > 5:
                    data['Rxx_x2'] = data[data_headings[3]] * VoverI    # data_headings 3 = Rxx_x
                    data['Rxx_y2'] = data[data_headings[4]] * VoverI    # data_headings 4 = Rxx_y
        data.sort_values(data_headings[0],0,inplace=True)

        # print(data)
        
    return data


def ComplexRotate(Raw_Re, Raw_Im, phase):
    Rotate = math.radians(phase)
    Rot_Re = Raw_Re*np.cos(Rotate) - Raw_Im*np.sin(Rotate)
    Rot_Im = Raw_Re*np.sin(Rotate) + Raw_Im*np.cos(Rotate)
    
    return Rot_Re, Rot_Im
    
def RofB(R_data,B_data,B_target):
    #Find value of R_data closest to corresponding B_target value in B_data
    min_val = np.amin(np.abs(B_data - B_target))
    min_index = indexOf(np.abs(B_data - B_target),min_val)  
    return R_data[min_index]

def BofR(R_data,B_data,R_target):
    #Find value of B_data closest to corresponding R_target value in R_data
    min_val = np.amin(np.abs(R_data - R_target))
    min_index = indexOf(np.abs(R_data - R_target),min_val)
    return B_data[min_index]

def apodize_data(data_struct,R_ind,order=1, background_mode="points",extra_point_inds=[],window_slices=[],start_point=0,chop_point=0,invert=False,show_plot=False):

    if "xx" in R_ind:
        R_dat = data_struct.Rxx_x
    elif "yy" in R_ind:
        R_dat = data_struct.Ryy_x
    elif "xy" in R_ind:
        R_dat = data_struct.Rxy_x
    elif "yx" in R_ind:
        R_dat = data_struct.Ryx_x
    elif "xx_grad" in R_ind:
        R_dat = data_struct.Rxx_grad

    # print(R_dat)
    
    if invert:
        B_pos = np.array(data_struct.An_Field[data_struct.An_Field < 0])[start_point:chop_point:-1]
        R_pos = np.array(R_dat[data_struct.An_Field < 0])[start_point:chop_point:-1]
        # print(R_pos,B_pos)

        B_0 = np.amax(B_pos)
        B_end = np.amin(B_pos)
        R_0 = RofB(R_pos,B_pos,B_0)
        R_end = RofB(R_pos,B_pos,B_end)

    else:
        B_pos = np.array(data_struct.An_Field[data_struct.An_Field > 0])[chop_point:start_point]
        R_pos = np.array(R_dat[data_struct.An_Field > 0])[chop_point:start_point]
        
        
        B_0 = np.amin(B_pos)
        B_end = np.amax(B_pos)
        # print(B_0,B_end)

        R_0 = RofB(R_pos,B_pos,B_0)

        R_end = RofB(R_pos,B_pos,B_end)


    if background_mode=="fit":
        # Using whole data set:
        raw_avg_fit = np.polyfit(np.array(B_pos),np.array(R_pos),order)
    
    
        avg_fit = np.pad(raw_avg_fit,(9-len(raw_avg_fit),0),mode="constant")
        # print(avg_fit)
        back_fun = (avg_fit[0]*(B_pos)**8 + avg_fit[1]*(B_pos)**7 + avg_fit[2]*(B_pos)**6 + avg_fit[3]*(B_pos)**5 + avg_fit[4]*(B_pos)**4 + 
                    avg_fit[5]*(B_pos)**3 + avg_fit[6]*(B_pos)**2 + avg_fit[7]*(B_pos)**1 + avg_fit[8]*(B_pos)**0)
        # lin_fit = [(R_end - R_0)/(B_end - B_0), R_0]


        if show_plot:
            plt.figure()
            plt.plot(B_pos,R_pos,color="b",label=r"$R_\mathrm{Long}$")
            plt.plot(B_pos, R_pos - back_fun,color="k")
            plt.plot(B_pos, back_fun,color="r",label=r"BackgroundFunction")
            plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
            plt.xlabel(r"$B$ (T)")
            plt.title(r"Resistance after 1st order background subtraction")
        # print(R_0,R_pos[0])
        # R_pos -= R_0
        # R_pos -= 0.9*lin_fit[0]*(B_pos-B_0)
        R_pos -= back_fun

    if background_mode=="points":
            
        if order == 0:

            if show_plot:
                plt.figure()
                plt.plot(B_pos,R_pos,color="b",label=r"R_\mathrm{Long}")
                plt.hlines(R_0,B_pos[-1],B_pos[0],color="r",label=r"Background Function")
                plt.title(r"Resistance after 0th order background subtraction")


            R_pos -= np.mean(R_pos)


        if order == 1:
            

                # lin_fit = np.polyfit((B_pos-B_0),R_pos - R_0,1)

                # lin_fit = [(R_end - R_0)/(B_end - B_0), R_0]


                if show_plot:
                    plt.figure()
                    plt.plot(B_pos,R_pos,color="b",label=r"$R_\mathrm{Long}$")
                    plt.plot(B_pos,(R_end - R_0)/(B_end - B_0)*(B_pos - B_0) + R_0,color="r",label=r"Background Function")
                    plt.plot(B_pos, R_pos - (R_end - R_0)/(B_end - B_0)*(B_pos - B_0) - R_0,color="k")
                    plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
                    plt.xlabel(r"$B$ (T)")
                    plt.title(r"Resistance after 1st order background subtraction")
                # print(R_0,R_pos[0])
                # R_pos -= R_0
                # R_pos -= 0.9*lin_fit[0]*(B_pos-B_0)
                R_pos -= (R_end - R_0)/(B_end - B_0)*(B_pos - B_0) + R_0

        if order == 2:

            # Using end points and determined middle point:
            raw_avg_fit = np.polyfit(np.array([B_0, B_pos[extra_point_inds], B_end]),np.array([R_0, R_pos[extra_point_inds], R_end]),deg=2) 

            # # Using whole data set:
            # raw_avg_fit = np.polyfit(np.array(B_pos- B_0),np.array(R_pos - R_0),2)
            
            
            avg_fit = np.pad(raw_avg_fit,(0,9-len(raw_avg_fit)),mode="constant")
            # print(avg_fit)

            if show_plot:
                plt.figure()
                plt.plot(B_pos,R_pos,color="b",label=r"$R_\mathrm{Hall}$")
                plt.plot(B_pos, raw_avg_fit[0] * (B_pos)**2 + raw_avg_fit[1] * (B_pos)**1 + raw_avg_fit[2] * (B_pos)**0,color="r",label=r"BackgroundFunction")
                # plt.plot(B_pos, avg_fit[0]*(B_pos - B_0)**8 + avg_fit[1]*(B_pos - B_0)**7 + avg_fit[2]*(B_pos - B_0)**6 + avg_fit[3]*(B_pos - B_0)**5 + avg_fit[4]*(B_pos - B_0)**4 +
                #          avg_fit[5]*(B_pos - B_0)**3 + avg_fit[6]*(B_pos - B_0)**2 + avg_fit[7]*(B_pos - B_0)**1 + avg_fit[8]*(B_pos - B_0)**0
                #          ,color="r",label=r"BackgroundFunction")
                plt.scatter(B_pos[extra_point_inds], R_pos[extra_point_inds])
                plt.legend()
                plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
                plt.xlabel(r"$B$ (T)")
                plt.title(r"Resistance after 2nd order background subtraction")

                # plt.figure()
                # # plt.plot(B_pos,R_pos-R_0,color="b",label=r"$R_\mathrm{Hall}$")
                # plt.plot(B_pos,R_pos - R_0 - (avg_fit[0]*(B_pos - B_0)**8 + avg_fit[1]*(B_pos - B_0)**7 + avg_fit[2]*(B_pos - B_0)**6 + avg_fit[3]*(B_pos - B_0)**5 + avg_fit[4]*(B_pos - B_0)**4 +
                #          avg_fit[5]*(B_pos - B_0)**3 + avg_fit[6]*(B_pos - B_0)**2 + avg_fit[7]*(B_pos - B_0)**1 + avg_fit[8]*(B_pos - B_0)**0)
                #          ,color="r",label=r"Post-Subtraction Function")
                # plt.legend()
                # plt.title(r"Resistance after 8th order Background Subtraction")

            R_pos -= raw_avg_fit[0] * (B_pos)**2 + raw_avg_fit[1] * (B_pos)**1 + raw_avg_fit[2] * (B_pos)**0


        if order == 8:

            raw_avg_fit = np.polyfit(np.array(B_pos- B_0),np.array(R_pos - R_0),3) # Try different weighting schemes
            avg_fit = np.pad(raw_avg_fit,(0,9-len(raw_avg_fit)),mode="constant")
            # print(len(avg_fit))


            if show_plot:
                plt.figure()
                plt.plot(B_pos,R_pos-R_0,color="b",label=r"$R_\mathrm{Hall}$")
                plt.plot(B_pos,avg_fit[0]*(B_pos - B_0)**8 + avg_fit[1]*(B_pos - B_0)**7 + avg_fit[2]*(B_pos - B_0)**6 + avg_fit[3]*(B_pos - B_0)**5 + avg_fit[4]*(B_pos - B_0)**4 +
                        avg_fit[5]*(B_pos - B_0)**3 + avg_fit[6]*(B_pos - B_0)**2 + avg_fit[7]*(B_pos - B_0)**1 + avg_fit[8]*(B_pos - B_0)**0
                        ,color="r",label=r"BackgroundFunction")
                plt.legend()
                plt.title(r"8th Order Background Subtraction Function")

                plt.figure()
                # plt.plot(B_pos,R_pos-R_0,color="b",label=r"$R_\mathrm{Hall}$")
                plt.plot(B_pos,R_pos - R_0 - (avg_fit[0]*(B_pos - B_0)**8 + avg_fit[1]*(B_pos - B_0)**7 + avg_fit[2]*(B_pos - B_0)**6 + avg_fit[3]*(B_pos - B_0)**5 + avg_fit[4]*(B_pos - B_0)**4 +
                        avg_fit[5]*(B_pos - B_0)**3 + avg_fit[6]*(B_pos - B_0)**2 + avg_fit[7]*(B_pos - B_0)**1 + avg_fit[8]*(B_pos - B_0)**0)
                        ,color="r",label=r"Post-Subtraction Function")
                plt.legend()
                plt.title(r"Resistance after 8th order Background Subtraction")

            R_pos -= R_0 + (avg_fit[0]*(B_pos - B_0)**8 + avg_fit[1]*(B_pos - B_0)**7 + avg_fit[2]*(B_pos - B_0)**6 + avg_fit[3]*(B_pos - B_0)**5 + avg_fit[4]*(B_pos - B_0)**4 +
                        avg_fit[5]*(B_pos - B_0)**3 + avg_fit[6]*(B_pos - B_0)**2 + avg_fit[7]*(B_pos - B_0)**1 + avg_fit[8]*(B_pos - B_0)**0)

    return R_pos, B_pos


def interpolate_data(R_pos,B_pos, invert=True, scaling_mode="linear", scaling_order=0, interp_ratio=10):
    '''Exponential Scaling order from Coleridge = 2.5'''
    # print(B_pos[np.abs(B_pos)==np.abs(np.amin(B_pos))])
    if invert:
        B_max = np.amax(B_pos)          ###NOTE TO THOMAS:
        B_min = np.amin(B_pos)             #Aren't these if, else statements doing the same thing?
    else:
        B_min = np.amin(B_pos)
        B_max = np.amax(B_pos)
    # print(B_min,B_pos[-1])
    
    # plt.figure()
    # plt.plot(np.linspace(1/B_max,1/B_min,interp_ratio*len(B_pos)))
    # plt.plot(B_pos)
    B_inv = 1/np.linspace(1/B_max,1/B_min,np.round(interp_ratio*len(B_pos)))
    if scaling_mode == "linear":
        scaling_fun = 1/np.abs(B_inv)**scaling_order
    
    elif scaling_mode == "exponential":
        scaling_fun = 0.5*np.exp(scaling_order/np.abs(B_inv))
            # print(R_pos,B_pos,B_inv)

    elif scaling_mode == "None":
        scaling_fun = 1


    if invert:
        # plt.figure()
        # plt.plot(np.flip(B_pos))
        # plt.plot(B_inv)
        R_interp = np.interp(B_inv,np.flip(B_pos), np.flip(R_pos))*scaling_fun
    else:
        # B_inv = np.flip(1/np.linspace(1/B_max,1/B_min,interp_ratio*len(B_pos)))
        R_interp = np.interp(B_inv,B_pos,R_pos)*scaling_fun

    # plt.figure()
    # plt.scatter(1/B_inv,R_interp)

    return R_interp, B_inv




def apod_NB(R_inv,B_inv,order=0,show_plot=False,invert=False):

    if order == 0:
        return R_inv, B_inv

    if order == 1: # Norton-Beer Weak Apodization

        C0 = 0.548
        C1 = -0.0833
        C2 = 0.5353
        C4 = 0
        label_str = r"NB1  Apodizing Function"


    if order == 2: # Norton-Beer Medium Apodization

        C0 = 0.26
        C1 = -0.154838
        C2 = 0.894838
        C4 = 0
        label_str = r"NB2  Apodizing Function"


    if order == 3: # Norton-Beer Medium Apodization

        C0 = 0.09
        C1 = 0
        C2 = 0.5875
        C4 = 0.3225
        label_str = r"NB3  Apodizing Function"


    fun_NB_1 = lambda x: C0*(1-x**2)**0 + C1*(1-x**2)**1 + C2*(1-x**2)**2 + C4*(1-x**2)**4
    if invert==False:
        norm_B_inv = (2*1/B_inv - 2*np.amin(1/B_inv))/(np.amax(1/B_inv)-np.amin(1/B_inv)) - 1
        apod_fun = fun_NB_1(norm_B_inv)

    if invert==True:
        norm_B_inv = (2*1/B_inv - 2*np.amin(1/B_inv))/(np.amax(1/B_inv)-np.amin(1/B_inv)) - 1
        # norm_B_inv = (1/B_inv - np.amin(1/B_inv))/(np.amax(1/B_inv)-np.amin(1/B_inv))
        # apod_fun = -1*fun_NB_1(norm_B_inv) + np.amin(fun_NB_1(norm_B_inv)) + 1
        apod_fun = fun_NB_1(norm_B_inv)
        # apod_fun = fun_NB_1(norm_B_inv)

    # print("Norm B range: ", norm_B_inv[0], norm_B_inv[-1])

    if show_plot:
        # plt.figure()
        # plt.plot(1/B_inv,R_inv,color="b",label=r"$R_\mathrm{Hall}$")
        # plt.title("")
        # plt.scatter(B_pos[extra_point_inds[0]],R_pos[extra_point_inds[0]])
        # plt.legend()

        plt.figure()
        plt.scatter(1/B_inv,np.amax(R_inv)*apod_fun,color="r",label=label_str)
        plt.plot(1/B_inv,R_inv,color="b",label=r"$R_\mathrm{Long}$")
        plt.legend()
        plt.title(r"Norton-Beer Apodization Function and Resistance Data")

        # plt.figure()
        # # plt.plot(1/B_inv,apod_fun,color="r",label=r"NB1  Apodizing Function")
        # plt.plot(1/B_inv,R_inv * apod_fun,color="b",label=r"$R_\mathrm{Long}$")
        # plt.legend()
        # plt.title(r"Resistance After Norton-Beer Apodization")



    return R_inv * apod_fun








if __name__ == "__main__":




    # Data analysis for D230831B_5 illuminated, 10-18-2023
    if 1==0:

        file_path = "Z:\\samples\\D230831B_5"
        Vg_vals = [-100]#[-200, -100, 0]
        for Vg in Vg_vals:
            file_name = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
            # print(file_names[i])
            D230831B_5_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])

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
            nu = 2
            rho_xx_par = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            rho_xy_par = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
            inv = pd.DataFrame({'B_field': D230831B_5_data.An_Field,
                                'Rxx': D230831B_5_data.Rxx_x,
                                'Rxy': D230831B_5_data.Rxy_x,
                                'p_xx_tot': rho_xx_tot,
                                'p_xy_tot': rho_xx_tot,
                                'p_det_tot': rho_det_tot,
                                })
            inv.sort_values(by='B_field',inplace=True,ignore_index=True)
            nu_bounds = []
            nu_bounds.append((0,0)) # nu = 0
            nu_bounds.append((0,0)) # nu = 1
            nu_bounds.append((3640,3700)) # nu = 2
            nu_bounds.append((1885,1895)) # nu = 3

            plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

            plt.figure()
            plt.plot(inv.B_field,inv.Rxx)
            plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="b",label=r"$\nu$ = 2")
            plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="r",label=r"$\nu$ = 3")
            plt.grid()
            plt.legend()

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


        rho_xx_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_xy_tot = D230831B_5_data.Rxx_x*(0.5/2.65)
        rho_det_tot = rho_xy_tot**2 + rho_xx_tot**2
        nu = 2
        rho_xx_par = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)
        rho_xy_par = rho_xx_tot * rho_det_tot /( (rho_xx_tot)**2 + (rho_xy_tot - rho_det_tot*nu*c.e**2/c.h)**2)








    # Attempt to create a contour plot of SdH curves for D230831B_6
    if 1==0:
        file_path = "Z:\\samples\\D230831B_6"
        Vg_vals = [0, 100, 200, 250, 300,350,400, 450, 500, 550]
        file_names = []
        i = 0
        for Vg in Vg_vals:
            file_names.append("D230831B_6_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat")
            # print(file_names[i])
            i+=1

        i=0
        for file in file_names:
            D230831B_6_data = get_dat_data(file_path,file,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])

            # windows: [start_point, chop_point]

            D230831B_6_data.Rxx_x = np.gradient(D230831B_6_data.Rxx_x) # First Deriv
            # D230831B_6_data.Rxx_x = np.gradient(D230831B_6_data.Rxx_x) # Second Deriv
            
            window_size = 4000
            # print(len(D230831B_6_data.Rxx_x))
            window = [(300+window_size),300]
            # print(len(D230831B_6_data.Rxx_x), window)
            

            D230831B_6_R_pos , D230831B_6_B_pos = apodize_data(D230831B_6_data,["xx"], order=0,background_mode="None",extra_point_inds=200, start_point=window[0],
                                                            chop_point = window[1],invert=False, show_plot=False)
            D230831B_6_R_inv , D230831B_6_B_inv = interpolate_data(D230831B_6_R_pos, D230831B_6_B_pos,
                                                                                                invert=False,scaling_order=1.5,scaling_mode="None")
            
            D230831B_6_R_inv = apod_NB(D230831B_6_R_inv,D230831B_6_B_inv,order=1,show_plot=False,invert=False)
            
            # D230831B_6_delt_B_inv_av = np.abs(1/D230831B_6_B_pos[0] - 1/D230831B_6_B_pos[-1])/(0.5*(len(D230831B_6_B_pos)-1))
            D230831B_6_delt_B_inv = 1/D230831B_6_B_inv[1:-1] - 1/D230831B_6_B_inv[0:-2]
            D230831B_6_delt_B_inv_av = np.mean(D230831B_6_delt_B_inv)
            n_points = 8*len(D230831B_6_R_inv)
            D230831B_6_trans = ft.rfft(D230831B_6_R_inv,n=n_points)
            D230831B_6_f_array =  np.arange(len(D230831B_6_trans)) / n_points / np.abs(D230831B_6_delt_B_inv_av) *c.e / c.h

            # Plot FFT's
            fft_start = 0#3520
            fft_cutoff = -1#-3520
            if i == 0:
                R_data = np.empty((len(Vg_vals),len(D230831B_6_data.An_Field)))
                spect_data = np.empty((len(Vg_vals),len(D230831B_6_f_array)))
                B_data = np.empty(len(D230831B_6_data.An_Field))
                R_data[i,:] = D230831B_6_data.Rxx_x#/np.max(D230831B_6_data.Rxx_x)
                B_data = D230831B_6_data.An_Field
                f_data = D230831B_6_f_array
                spect_data[i,:] = np.abs(D230831B_6_trans)/np.amax(np.abs(D230831B_6_trans))
            else: 
                R_data[i,:] = np.interp(B_data[:],D230831B_6_data.An_Field,D230831B_6_data.Rxx_x)
                spect_data[i,:] = np.interp(f_data[:],D230831B_6_f_array,np.abs(D230831B_6_trans)/np.amax(np.abs(D230831B_6_trans)))
            i+= 1
        
        B_max = 4
        B_min = 0.3
        # np.where((B_data>0)&(B_data<B_max),B_data,1e-6)
        R_data = R_data[:,(B_data>B_min)&(B_data<B_max)]
        R_data -= np.amin(R_data)
        B_data = B_data[(B_data>B_min)&(B_data<B_max)]# np.where((B_data>0)&(B_data<B_max),R_data,1e-6)
        
        

        # # plt.subplots(1,2)
        # # plt.subplot(1,2,1)
        # plt.figure()
        # level_array = np.geomspace(np.amin(R_data)+1,1.0*np.amax(R_data),1000)
        # plt.contourf(1/B_data,Vg_vals,R_data,levels=level_array)
        # # plt.xlim(1/B_max,2)
        # plt.xlabel(r"1/B (T$^{-1}$)")
        # plt.ylabel(r"$V_\mathrm{g}$ (mV)")
        # plt.title("SdH Derivative Contour")

        file_path = "Z:\\samples\\D230831B_6"
        file_name = "16Oct2023_154207.dat"

        os.chdir(file_path)
        data_headings=['Vg', 'Rxx_x', 'col_3','col_4', 'col_5', 'Rxy_x', 'col_6',  'col_7','col_8',
                                                      'col_9', 'col_10', 'col_11', 'col_12', 'col_13','col_14']
        D230831B_6_Vsweep_data = pd.read_table(file_name,delimiter="\t",header=None,names=data_headings,index_col=1)

        n = 0.1*1e-6/(D230831B_6_Vsweep_data.Rxy_x*c.e)#*(2.65/0.5)
        V_list = D230831B_6_Vsweep_data.Vg

        # plt.subplot(1,2,2)
        plt.figure()
        level_array = np.geomspace(np.amin(np.abs(spect_data))+0.001,1.0*np.amax(np.abs(spect_data)),1000)
        plt.contourf(1e-4*f_data,Vg_vals,np.abs(spect_data),levels=level_array)
        # plt.hold(True)
        plt.plot(1e-4*n,1e3*V_list,zorder=1,color="red")
        # plt.xlim(1/B_max,2)
        plt.xlabel(r"n (cm$^{-2}$)")
        plt.ylabel(r"$V_\mathrm{g}$ (mV)")
        plt.xlim(0,6e11)
        plt.ylim(np.amin(Vg_vals),np.amax(Vg_vals))
        plt.title("D230831B_6 SdH Derivative FFT Contour")



    

    # Attempt to create a contour plot of SdH curves for D230831B_5
    if 1==0:
        file_path = "D230831B 2nd cooldown\Full Sweeps"
        Vg_vals = [-200, -100, 0, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]#, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750,800]
        file_names = []
        i = 0
        for Vg in Vg_vals:
            file_names.append("D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat")
            # print(file_names[i])
            i+=1

        i=0
        for file in file_names:
            D230831B_5_data = get_dat_data(file_path,file,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
            # scaling_1 = 80e-4
            # knee = 800
            # scaling_2 = 7e-4
            # if Vg <=knee:
            #     new_B = D230831B_5_data.An_Field / (scaling_1*Vg + 1)
            # else:
            #     new_B = D230831B_5_data.An_Field / (scaling_2*(Vg - knee) + scaling_1*knee + 1)

            # D230831B_5_data.An_Field = new_B

            # windows: [start_point, chop_point]
            # window = [-1,0] # Full curve
            # window = [-305,3530] # SdH Oscillations
            # window = [-305,4030] # Spin-splitting oscillations
            # window = [-515,3500] # Non-split oscillations and higher-order peaks

            D230831B_5_data.Rxx_x = np.gradient(D230831B_5_data.Rxx_x,D230831B_5_data.An_Field) # First Deriv
            # D230831B_5_data.Rxx_x = np.gradient(D230831B_5_data.Rxx_x) # Second Deriv
            
            window_size = 4000
            # print(len(D230831B_5_data.Rxx_x))
            window = [(300+window_size),300]
            # print(len(D230831B_5_data.Rxx_x), window)
            

            D230831B_5_R_pos , D230831B_5_B_pos = apodize_data(D230831B_5_data,["xx"], order=0,background_mode="None",extra_point_inds=200, start_point=window[0],
                                                            chop_point = window[1],invert=False, show_plot=False)
            D230831B_5_R_inv , D230831B_5_B_inv = interpolate_data(D230831B_5_R_pos, D230831B_5_B_pos,
                                                                                                invert=False,scaling_order=1.5,scaling_mode="None")
            
            D230831B_5_R_inv = apod_NB(D230831B_5_R_inv,D230831B_5_B_inv,order=1,show_plot=False,invert=False)
            
            # D230831B_5_delt_B_inv_av = np.abs(1/D230831B_5_B_pos[0] - 1/D230831B_5_B_pos[-1])/(0.5*(len(D230831B_5_B_pos)-1))
            D230831B_5_delt_B_inv = 1/D230831B_5_B_inv[1:-1] - 1/D230831B_5_B_inv[0:-2]
            D230831B_5_delt_B_inv_av = np.mean(D230831B_5_delt_B_inv)
            n_points = 8*len(D230831B_5_R_inv)
            D230831B_5_trans = ft.rfft(D230831B_5_R_inv,n=n_points)
            D230831B_5_f_array =  np.arange(len(D230831B_5_trans)) / n_points / np.abs(D230831B_5_delt_B_inv_av) *c.e / c.h

            # Plot FFT's
            fft_start = 0#3520
            fft_cutoff = -1#-3520
            if i == 0:
                R_data = np.empty((len(Vg_vals),len(D230831B_5_data.An_Field)))
                spect_data = np.empty((len(Vg_vals),len(D230831B_5_f_array)))
                B_data = np.empty(len(D230831B_5_data.An_Field))
                R_data[i,:] = D230831B_5_data.Rxx_x#/np.max(D230831B_5_data.Rxx_x)
                B_data = D230831B_5_data.An_Field
                f_data = D230831B_5_f_array
                spect_data[i,:] = np.abs(D230831B_5_trans)/np.amax(np.abs(D230831B_5_trans))
            else: 
                R_data[i,:] = np.interp(B_data[:],D230831B_5_data.An_Field,D230831B_5_data.Rxx_x)
                spect_data[i,:] = np.interp(f_data[:],D230831B_5_f_array,np.abs(D230831B_5_trans)/np.amax(np.abs(D230831B_5_trans)))
            i+= 1
        
        B_max = 4
        B_min = 0.3
        # np.where((B_data>0)&(B_data<B_max),B_data,1e-6)
        R_data = R_data[:,(B_data>B_min)&(B_data<B_max)]
        R_data -= np.amin(R_data)
        B_data = B_data[(B_data>B_min)&(B_data<B_max)]# np.where((B_data>0)&(B_data<B_max),R_data,1e-6)
        
        

        # # plt.subplots(1,2)
        # # plt.subplot(1,2,1)
        # plt.figure()
        # level_array = np.geomspace(np.amin(R_data)+1,1.0*np.amax(R_data),1000)
        # plt.contourf(1/B_data,Vg_vals,R_data,levels=level_array)
        # # plt.xlim(1/B_max,2)
        # plt.xlabel(r"1/B (T$^{-1}$)")
        # plt.ylabel(r"$V_\mathrm{g}$ (mV)")
        # plt.title("SdH Derivative Contour")

        file_path = "Z:\\samples\\D230831B_5\\Measurements 9-10-2023"
        file_name = "09Oct2023_130204.dat"

        os.chdir(file_path)
        data_headings=['Vg', 'Rxx_x', 'col_3','col_4', 'col_5', 'Rxy_x', 'col_6',  'col_7','col_8',
                                                      'col_9', 'col_10', 'col_11', 'col_12', 'col_13','col_14']
        D230831B_5_Vsweep_data = pd.read_table(file_name,delimiter="\t",header=None,names=data_headings,index_col=1)

        n = -0.1*1e-6/(D230831B_5_Vsweep_data.Rxy_x*c.e)#*(2.7/0.5)
        V_list = D230831B_5_Vsweep_data.Vg

        # plt.subplot(1,2,2)
        plt.figure()
        level_array = np.geomspace(np.amin(np.abs(spect_data))+0.001,1.0*np.amax(np.abs(spect_data)),1000)
        plt.contourf(1e-4*f_data,Vg_vals,np.abs(spect_data),levels=level_array)
        # plt.hold(True)
        plt.plot(1e-4*n,1e3*V_list,zorder=1,color="red")
        # plt.xlim(1/B_max,2)
        plt.xlabel(r"n (cm$^{-2}$)")
        plt.ylabel(r"$V_\mathrm{g}$ (mV)")
        plt.xlim(0,6e11)
        plt.ylim(np.amin(Vg_vals),np.amax(Vg_vals))
        plt.title("D230831B_5 SdH Derivative FFT Contour")




    # Data analysis for D230831B_5 dark, 10.102023
    if 1==0:


        # file_path = "Z:\\samples\\D230831B_5"
        # file_name = "D230831B_5_inv_Bsweep_675mV_Vg.dat"
        # Vg_vals = 675

        file_path = "Z:\\samples\\D230831B_5"
        Vg_vals = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800]#[-100, 0, 75, 100,125,  200, 250, 300,  400]# [-200, -100, 0, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
        # file_names.append("D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat")
        plt.figure()
        for Vg in Vg_vals:
            file_name = "D230831B_5_inv_Bsweep_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"

            D230831B_5_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
            # D230831B_5_data.dropna(inplace=True)
            # print(D230831B_5_data)

            # windows: [start_point, chop_point]
            # window = [-1,0] # Full curve
            # window = [-305,3530] # SdH Oscillations
            # window = [-305,4030] # Spin-splitting oscillations
            # window = [-515,3500] # Non-split oscillations and higher-order peaks
        
            D230831B_5_data.Rxx_x = np.gradient(D230831B_5_data.Rxx_x)
            window_size = 1200
            window = [len(D230831B_5_data.Rxx_x)-(300+window_size),300]

            D230831B_5_R_pos , D230831B_5_B_pos = apodize_data(D230831B_5_data,["xx"], order=0,background_mode="fit",extra_point_inds=200, start_point=window[0],
                                                            chop_point = window[1],invert=False, show_plot=False)
            D230831B_5_R_inv , D230831B_5_B_inv = interpolate_data(D230831B_5_R_pos, D230831B_5_B_pos,
                                                                                            invert=False,scaling_order=2,scaling_mode="None")
            # print(len(D230831B_5_R_pos))
            # D230831B_5_R_inv = apod_NB(D230831B_5_R_inv,D230831B_5_B_inv,order=1,show_plot=True,invert=False)
            # print(D230831B_5_delt_B)

            # # Plot raw data
            # # plt.figure()
            # plt.subplots(1,2)
            
            # plt.subplot(1,2,1)
            # # plt.plot(D230831B_5_data.An_Field,np.abs(D230831B_5_data.Rxx_x + 1.j *D230831B_5_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
            # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxx_x ,color="b",label=r"Rxx_x")
            # R_range_start = 0
            # R_rang_stop = -500
            # plt.scatter(D230831B_5_data.An_Field[indexOf(D230831B_5_data.Rxx_x,np.amin(D230831B_5_data.Rxx_x[R_range_start:R_rang_stop]))],np.amin(D230831B_5_data.Rxx_x[R_range_start:R_rang_stop]))
            # plt.annotate(np.format_float_positional(np.amin(D230831B_5_data.Rxx_x[R_range_start:R_rang_stop]), unique = False, precision=1)+ r" $\Omega$",[1.05*D230831B_5_data.An_Field[indexOf(D230831B_5_data.Rxx_x,np.amin(D230831B_5_data.Rxx_x[R_range_start:R_rang_stop]))],1.1*np.amin(D230831B_5_data.Rxx_x[R_range_start:R_rang_stop])])
            # # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxy_x,color="r",label=r"Rxy_x")
            # # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
            # # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
            # # plt.vlines(D230831B_5_B_pos[0],-0.1,1.2*np.amax(D230831B_5_data.Rxx_x),color="r",linestyle="dashed")
            # # plt.vlines(D230831B_5_B_pos[-1],-0.1,1.2*np.amax(D230831B_5_data.Rxx_x),color="r",linestyle="dashed")
            # plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
            # plt.xlabel(r"$B$ (T)")
            # plt.title(r"Long. Resistance, $T$ = 1 K, sample D230831B_5, $V_\mathrm{g}$ = " + np.format_float_positional(Vg,precision=4,trim='-') + ' mV')
            # # plt.ylim(0,5000)
            # # plt.xlim(-4,0)
            # # plt.legend()


            # plt.subplot(1,2,2)
            # plt.plot(D230831B_5_data.An_Field,np.abs(D230831B_5_data.Rxx_x + 1.j *D230831B_5_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
            # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxx_x ,color="b",label=r"Rxx_x")
            plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxy_x,label=r"$V_\mathrm{g}$=" + np.format_float_positional(Vg,precision=4,trim='-') + ' mV')
            # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
            # plt.plot(D230831B_5_data.An_Field,D230831B_5_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
            # plt.vlines(D230831B_5_B_pos[0],-0.1,1.2*np.amax(D230831B_5_data.Rxx_x),color="r",linestyle="dashed")
            # plt.vlines(D230831B_5_B_pos[-1],-0.1,1.2*np.amax(D230831B_5_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xy}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Hall Resistance, sample D230831B_5, T = 1 K")
        # plt.ylim(0,18000)
            # plt.xlim(-4,0)
        plt.legend()


        # Plot Inverted Data after apodization and interpolation
        # plt.figure()
        # plt.plot(1/D230831B_5_B_inv,D230831B_5_R_inv)
        # plt.xlabel(r"$1/B$ (T$^{-1}$)")
        # plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        # plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D230831B_5_delt_B_inv = 1/D230831B_5_B_inv[1:-1] - 1/D230831B_5_B_inv[0:-2]
        D230831B_5_delt_B_inv_av = np.mean(D230831B_5_delt_B_inv)

        n_points = 8*len(D230831B_5_R_inv)
        D230831B_5_trans = ft.rfft(D230831B_5_R_inv,n=n_points)
        D230831B_5_f_array =  np.arange(len(D230831B_5_trans)) / n_points / np.abs(D230831B_5_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = -1#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_5_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_5_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        # plt.figure()
        plt.subplots(1,2)
        peaks = sig.find_peaks(1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff]), 
                               height = 0.1*np.amax(1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff])))
        # print(peaks)
        # peak_density = D230831B_5_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831B_5_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831B_5_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        plt.subplot(1,2,1)
        plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        for peak in peaks[0]:
            plt.scatter(1e-4*D230831B_5_f_array[fft_start+peak],1e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff])[peak])
            plt.annotate(np.format_float_scientific(1e-4*D230831B_5_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.05e-4*D230831B_5_f_array[fft_start+peak],0.9e-6*np.abs(D230831B_5_trans[fft_start:fft_cutoff])[peak]])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831B_5, $V_\mathrm{g}$ = ' + np.format_float_positional(675,precision=4,trim='-') + ' mV')
        plt.xlim(0,6e11)

        plt.subplot(1,2,2)
        plt.plot(1/D230831B_5_B_inv,D230831B_5_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"d$R_\mathrm{xx}$d/$B$ ($\Omega T^{-1}$)")
        plt.title(r"Long. Resistance vs 1/B, after processing")

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831B_5_f_array[fft_start:fft_cutoff],np.angle(D230831B_5_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')









    # Data analysis for D230831B_5 illuminated, 09.27.2023
    if 1==0:


        file_path = "Z:\\User\\Thomas\\D230831B_ref"
        file_name = "D230831B_ref_SdH1586_ill1K.dat"
        D230831B_ref_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=False,data_headings=["An_Field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
        # D230831B_ref_data.dropna(inplace=True)
        # print(D230831B_ref_data)

        # windows: [start_point, chop_point]
        # window = [-1,0] # Full curve
        # window = [-1130,170] # SdH Oscillations, high B, lots of beating, use order = 1
        # window = [-475,2050] # Spin-splitting oscillations, use order = 2, extra_point_inds = 200
        window_size = 606
        # bckgrnd_window = [-475,370]#
        # start_window = [bckgrnd_window[0]+399,2050-bckgrnd_window[1]]
        start_window = [-475,2050]#[-475,370]# 
        window_ind = 0
        window_stop_ind = 20
        window_increment = 50
        window = [start_window[0]-window_ind*window_increment,start_window[1]+window_ind*window_increment] # Spectogram window
        # print(window_size/2)
        # D230831B_ref_R_spec_array = np.empty((window_stop_ind+1,round(window_size/10)))
        # D230831B_ref_B_spec_array = np.empty((window_stop_ind+1,round(window_size/10)))
        # D230831B_ref_R_pos_full , D230831B_ref_B_pos_full = apodize_data(D230831B_ref_data,["xx"], order=4,background_mode="fit",extra_point_inds=200, start_point=bckgrnd_window[0],
        #                                                 chop_point = bckgrnd_window[1],invert=True, show_plot=False) # Full data window
        
        D230831B_ref_data.Rxx_x = np.gradient(D230831B_ref_data.Rxx_x)

        print(start_window)
        for i in range(window_stop_ind):
            
            window_ind = i
            window = [start_window[0]-window_ind*window_increment,start_window[1]-window_ind*window_increment] # Spectogram window
            # window = [start_window[0]+window_ind*window_increment,start_window[1]+window_ind*window_increment]

            D230831B_ref_R_pos , D230831B_ref_B_pos = apodize_data(D230831B_ref_data,["xx"], order=0,background_mode="fit",extra_point_inds=200, start_point=window[0],
                                                        chop_point = window[1],invert=True, show_plot=True)
            
            # print(window[1],window[0])
            # D230831B_ref_R_pos = D230831B_ref_R_pos_full[window[1]:window[0]]
            # D230831B_ref_B_pos = D230831B_ref_B_pos_full[window[1]:window[0]]

            # print("B range: ",D230831B_ref_B_pos[0]," - ",D230831B_ref_B_pos[-1])
            D230831B_ref_R_inv , D230831B_ref_B_inv = interpolate_data(D230831B_ref_R_pos, D230831B_ref_B_pos,
                                                                                            invert=True,scaling_order=2,scaling_mode="None")
            # print(len(D230831B_ref_R_pos))
            # D230831B_ref_R_inv = apod_NB(D230831B_ref_R_inv,D230831B_ref_B_inv,order=3,show_plot=False,invert=False)
            # D230831B_ref_R_spec_array[window_ind,:] = D230831B_ref_R_inv
            # D230831B_ref_B_spec_array[window_ind,:] = D230831B_ref_B_inv

            D230831B_ref_delt_B_inv = 1/D230831B_ref_B_inv[1:-1] - 1/D230831B_ref_B_inv[0:-2]
            D230831B_ref_delt_B_inv_av = np.mean(D230831B_ref_delt_B_inv)

            test_delt_B = np.abs(1/D230831B_ref_B_pos[0] - 1/D230831B_ref_B_pos[-1])/(0.5*(len(D230831B_ref_B_pos)-1))

            print(D230831B_ref_delt_B_inv_av, test_delt_B)

            D230831B_ref_delt_B_inv_av = test_delt_B
            # plt.figure()
            # plt.plot(D230831B_ref_delt_B_inv)

            n_points = 4*len(D230831B_ref_R_inv)
            D230831B_ref_trans = ft.rfft(D230831B_ref_R_inv,n=n_points)
            # print("Check: ", len(D230831B_ref_trans), n_points)
            D230831B_ref_f_array =  np.arange(len(D230831B_ref_trans)) / n_points / np.abs(D230831B_ref_delt_B_inv_av) *c.e / c.h

            # Plot FFT's
            fft_start = 0#3520
            fft_cutoff = 150#-3520
            # plt.subplots(2,1)

            # plt.subplot(2,1,1)
            # plt.plot(D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_ref_trans[fft_start:fft_cutoff]))
            # plt.xlabel(r"$f_{1/B}$ (T)")
            # plt.ylabel(r'Real Amplitude')
            # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

            # plt.subplot(2,1,2)
            # plt.plot(D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_ref_trans[fft_start:fft_cutoff]))
            # plt.xlabel(r"$f_{1/B}$ (T)")
            # plt.ylabel(r'Imaginary Amplitude')

            # plt.figure(figsize=[1,2])
            plt.subplots(1,3,figsize=[13,4])
            
            peaks = sig.find_peaks(1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff]))
            # print(peaks)
            # peak_density = D230831B_ref_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831B_ref_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831B_ref_trans[fft_start:fft_cutoff])))]
            # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
            plt.subplot(1,3,1)
            plt.plot(1e-4*D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff]))
            for peak in peaks[0]:
                plt.scatter(1e-4*D230831B_ref_f_array[fft_start+peak],1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff])[peak])
                plt.annotate(np.format_float_scientific(1e-4*D230831B_ref_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.05e-4*D230831B_ref_f_array[fft_start+peak],0.95e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff])[peak]])
            plt.ylabel(r'FFT Amplitude')
            plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
            plt.xlim(0,5e11)
            plt.title(r'FFT in 1/B')

            plt.subplot(1,3,3)
            # # Plot raw data
            plt.plot(D230831B_ref_B_pos, D230831B_ref_R_pos ,color="b",label=r"Rxx_x")

            # plt.vlines(D230831B_ref_B_pos[0],-0.1,1.2*np.amax(D230831B_ref_data.Rxx_x),color="r",linestyle="dashed")
            # plt.vlines(D230831B_ref_B_pos[-1],-0.1,1.2*np.amax(D230831B_ref_data.Rxx_x),color="r",linestyle="dashed")
            plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
            plt.xlabel(r"$B$ (T)")
            plt.title(r"Long. Resistance, $T$ = 1 K, sample D230831B_ref")
            # # plt.ylim(0,400)
            # # plt.xlim(-4,0)
            # plt.legend()

            plt.subplot(1,3,2)
            # Plot Inverted Data after apodization and interpolation
            plt.plot(1/D230831B_ref_B_inv,D230831B_ref_R_inv)
            plt.xlabel(r"$1/B$ (T$^{-1}$)")
            plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
            plt.title(r"Long. Resistance vs 1/B, after processing")


        # print(D230831B_ref_delt_B)

        # def variable_apod(R_array):
        #     R_out = np.zeros(np.shape(R_array))
        #     for R_ind, R_dat  in enumerate(R_array):
        #         R_dat = sig.detrend(R_dat)
        #         R_out[R_ind] = apod_NB(R_dat,1/np.linspace(1,-1,len(R_dat)),order=2,show_plot=False,invert=True)
        #     return R_out

        # Plot raw data
        plt.figure()
        # plt.plot(D230831B_ref_data.An_Field,np.abs(D230831B_ref_data.Rxx_x + 1.j *D230831B_ref_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        plt.vlines(D230831B_ref_B_pos[0],-0.1,1.2*np.amax(D230831B_ref_data.Rxx_x),color="r",linestyle="dashed")
        plt.vlines(D230831B_ref_B_pos[-1],-0.1,1.2*np.amax(D230831B_ref_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 1 K, sample D230831B_ref")
        # plt.ylim(0,400)
        # plt.xlim(-4,0)
        plt.legend()


        # # Plot Inverted Data after apodization and interpolation
        # plt.figure()
        # plt.plot(1/D230831B_ref_B_inv,D230831B_ref_R_inv)
        # plt.xlabel(r"$1/B$ (T$^{-1}$)")
        # plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        # plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        # D230831B_ref_delt_B_inv = 1/D230831B_ref_B_inv[1:-1] - 1/D230831B_ref_B_inv[0:-2]
        # D230831B_ref_delt_B_inv_av = np.mean(D230831B_ref_delt_B_inv)
        # fft_window_size = 200
        # D230831B_ref_f_array, D230831B_ref_B_inv_array, D230831B_ref_spectrogram = sig.spectrogram(D230831B_ref_R_inv,fs=1/ np.abs(D230831B_ref_delt_B_inv_av),
        #                                                                                            window="boxcar",nperseg=fft_window_size,nfft=8*fft_window_size,
        #                                                                                            noverlap=0.9*fft_window_size,detrend=variable_apod)
        # # print(len(D230831B_ref_f_array))
        # # D230831B_ref_f_array = np.arange(fft_window_size) / (fft_window_size*8) / np.abs(D230831B_ref_delt_B_inv_av) *c.e / c.h
        # # sig.spectrogram()
        # fft_start_f = 0
        # fft_stop_f = 300
        
        # for i in range(len(D230831B_ref_spectrogram[0,:])):
        #     D230831B_ref_spectrogram[:,i] = D230831B_ref_spectrogram[:,i]/np.amax(D230831B_ref_spectrogram[:,i])

        # plt.figure()
        # level_array = np.geomspace(np.amin(D230831B_ref_spectrogram[fft_start_f:fft_stop_f,:]),1*np.amax(D230831B_ref_spectrogram[fft_start_f:fft_stop_f,:]),800)
        # plt.contour(D230831B_ref_B_inv_array, D230831B_ref_f_array[fft_start_f:fft_stop_f], D230831B_ref_spectrogram[fft_start_f:fft_stop_f,:],levels=level_array)
        # plt.xlabel(r"B")
        # plt.ylabel(r"Frequency")


        D230831B_ref_delt_B_inv = 1/D230831B_ref_B_inv[1:-1] - 1/D230831B_ref_B_inv[0:-2]
        D230831B_ref_delt_B_inv_av = np.mean(D230831B_ref_delt_B_inv)

        n_points = 8*len(D230831B_ref_R_inv)
        D230831B_ref_trans = ft.rfft(D230831B_ref_R_inv,n=n_points)
        # print("Check: ", len(D230831B_ref_trans), n_points)
        D230831B_ref_f_array =  np.arange(len(D230831B_ref_trans)) / n_points / np.abs(D230831B_ref_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 150#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_ref_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_ref_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        # plt.figure()
        # # plt.subplots(2,1)
        
        # peaks = sig.find_peaks(1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff]), height = 1e-3)
        # print(peaks)
        # # peak_density = D230831B_ref_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831B_ref_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831B_ref_trans[fft_start:fft_cutoff])))]
        # # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # # plt.subplot(2,1,1)
        # plt.plot(1e-4*D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # for peak in peaks[0]:
        #     plt.scatter(1e-4*D230831B_ref_f_array[fft_start+peak],1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff])[peak])
        #     plt.annotate(np.format_float_scientific(1e-4*D230831B_ref_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.05e-4*D230831B_ref_f_array[fft_start+peak],0.95e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff])[peak]])
        # plt.ylabel(r'FFT Amplitude')
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831B_ref')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831B_ref_f_array[fft_start:fft_cutoff],np.angle(D230831B_ref_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')







    # Data analysis for F150709B illuminated, 09.28.2023
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\ETH Zurich Materials"
        file_name = "F150709B_ill.txt"
        F150709B_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True, data_headings=["variable x","lockin2 x", "lockin2 y","lockin3 x", "lockin3 y"])
        # F150709B_data.dropna(inplace=True)
        # print(F150709B_data)

        # windows: [start_point, chop_point]
        # window = [-1,0] # Full curve
        # window = [-1500,710] # SdH Oscillations, high B, lots of beating
        # window = [-3130,208] # Low B, Spin-splitting oscillations,  use order = 2, extra_point_inds = 308
        window = [-3430,20] # Very Low B, Spin-splitting oscillations


        F150709B_R_pos , F150709B_B_pos = apodize_data(F150709B_data,["xx"], order=1,extra_point_inds=428, start_point=window[0],chop_point = window[1],invert=False, show_plot=True) # Full data window
        F150709B_R_inv , F150709B_B_inv , F150709B_delt_B = interpolate_data(F150709B_R_pos, F150709B_B_pos,invert=False,scaling_order=1)
        # print(len(F150709B_R_pos))
        # F150709B_R_inv = apod_NB(F150709B_R_inv,F150709B_B_inv,order=1,show_plot=True,invert=False)
        # print(F150709B_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(F150709B_data.An_Field,np.abs(F150709B_data.Rxx_x + 1.j *F150709B_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(F150709B_data.An_Field,F150709B_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(F150709B_data.An_Field,F150709B_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(F150709B_data.An_Field,F150709B_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(F150709B_data.An_Field,F150709B_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        # plt.vlines(F150709B_B_pos[0],0,1.2*np.amax(F150709B_data.Rxx_x),color="r",linestyle="dashed")
        # plt.vlines(F150709B_B_pos[-1],0,1.2*np.amax(F150709B_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 1 K, sample F150709B")
        plt.ylim(0,10e-5)
        plt.xlim(0,1)
        plt.legend()


        # # Plot Inverted Data after apodization and interpolation
        # plt.figure()
        # plt.plot(1/F150709B_B_inv,F150709B_R_inv)
        # plt.xlabel(r"$1/B$ (T$^{-1}$)")
        # plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        # plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        # F150709B_delt_B_inv = 1/F150709B_B_inv[1:-1] - 1/F150709B_B_inv[0:-2]
        # F150709B_delt_B_inv_av = np.mean(F150709B_delt_B_inv)

        # n_points = 8*len(F150709B_R_inv)
        # F150709B_trans = ft.rfft(F150709B_R_inv,n=n_points)
        # F150709B_f_array =  np.arange(len(F150709B_trans)) / n_points / np.abs(F150709B_delt_B_inv_av) *c.e / c.h

        # # Plot FFT's
        # fft_start = 0#3520
        # fft_cutoff = 250#-3520
        # # # plt.subplots(2,1)

        # # plt.subplot(2,1,1)
        # # plt.plot(F150709B_f_array[fft_start:fft_cutoff],1e-6*np.real(F150709B_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # # plt.ylabel(r'Real Amplitude')
        # # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # # plt.subplot(2,1,2)
        # # plt.plot(F150709B_f_array[fft_start:fft_cutoff],1e-6*np.imag(F150709B_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # # plt.ylabel(r'Imaginary Amplitude')

        # plt.figure()
        # # plt.subplots(2,1)
        
        # peaks = sig.find_peaks(1e-6*np.abs(F150709B_trans[fft_start:fft_cutoff]), prominence=(0.3e-8,None),width=0.5e1)
        # print(peaks)
        # # peak_density = F150709B_f_array[fft_start:fft_cutoff][indexOf(np.abs(F150709B_trans[fft_start:fft_cutoff]),np.amax(np.abs(F150709B_trans[fft_start:fft_cutoff])))]
        # # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # # plt.subplot(2,1,1)
        # plt.plot(1e-4*F150709B_f_array[fft_start:fft_cutoff],1e-6*np.abs(F150709B_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # for peak in peaks[0]:
        #     plt.scatter(1e-4*F150709B_f_array[fft_start+peak],1e-6*np.abs(F150709B_trans[fft_start:fft_cutoff])[peak])
        #     plt.annotate(np.format_float_scientific(1e-4*F150709B_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0),[1.0e-4*F150709B_f_array[fft_start+peak],1.02e-6*np.abs(F150709B_trans[fft_start:fft_cutoff])[peak]])
        # plt.ylabel(r'FFT Amplitude')
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.title(r'FFT in 1/B of Processed Long. Resistance, sample F150709B')

        # # plt.subplot(2,1,2)
        # # plt.plot(1e-4*F150709B_f_array[fft_start:fft_cutoff],np.angle(F150709B_trans[fft_start:fft_cutoff]))
        # # # plt.xlabel(r"$f_{1/B}$ (T)")
        # # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # # plt.ylabel(r'Phase (rads)')
    






    # Data analysis for D181022A illuminated, 09.28.2023
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\ETH Zurich Materials"
        file_name = "D181022A ill.txt"
        D181022A_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],has_header=True, data_headings=["variable x","lockin2 x", "lockin2 y","lockin3 x", "lockin3 y"])
        # D181022A_data.dropna(inplace=True)
        # print(D181022A_data)

        # windows: [start_point, chop_point]
        window = [-1,0] # Full curve
        # window = [-1500,710] # SdH Oscillations, high B, lots of beating
        # window = [-3130,208] # Low B, Spin-splitting oscillations,  use order = 2, extra_point_inds = 308


        D181022A_R_pos , D181022A_B_pos = apodize_data(D181022A_data,["xx"], order=2,extra_point_inds=428, start_point=window[0],chop_point = window[1],invert=False, show_plot=True) # Full data window
        D181022A_R_inv , D181022A_B_inv , D181022A_delt_B = interpolate_data(D181022A_R_pos, D181022A_B_pos,invert=False,scaling_order=1)
        print(len(D181022A_R_pos))
        # D181022A_R_inv = apod_NB(D181022A_R_inv,D181022A_B_inv,order=1,show_plot=True,invert=False)
        # print(D181022A_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(D181022A_data.An_Field,np.abs(D181022A_data.Rxx_x + 1.j *D181022A_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D181022A_data.An_Field,D181022A_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D181022A_data.An_Field,D181022A_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D181022A_data.An_Field,D181022A_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D181022A_data.An_Field,D181022A_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        plt.vlines(D181022A_B_pos[0],0,1.2*np.amax(D181022A_data.Rxx_x),color="r",linestyle="dashed")
        plt.vlines(D181022A_B_pos[-1],0,1.2*np.amax(D181022A_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 1 K, sample D181022A")
        plt.ylim(0,6e-4)
        plt.xlim(0,2)
        plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/D181022A_B_inv,D181022A_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D181022A_delt_B_inv = 1/D181022A_B_inv[1:-1] - 1/D181022A_B_inv[0:-2]
        D181022A_delt_B_inv_av = np.mean(D181022A_delt_B_inv)

        n_points = 8*len(D181022A_R_inv)
        D181022A_trans = ft.rfft(D181022A_R_inv,n=n_points)
        D181022A_f_array =  np.arange(len(D181022A_trans)) / n_points / np.abs(D181022A_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 250#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D181022A_f_array[fft_start:fft_cutoff],1e-6*np.real(D181022A_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D181022A_f_array[fft_start:fft_cutoff],1e-6*np.imag(D181022A_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peaks = sig.find_peaks(1e-6*np.abs(D181022A_trans[fft_start:fft_cutoff]), prominence=(0.3e-8,None),width=0.5e1)
        print(peaks)
        # peak_density = D181022A_f_array[fft_start:fft_cutoff][indexOf(np.abs(D181022A_trans[fft_start:fft_cutoff]),np.amax(np.abs(D181022A_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*D181022A_f_array[fft_start:fft_cutoff],1e-6*np.abs(D181022A_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        for peak in peaks[0]:
            plt.scatter(1e-4*D181022A_f_array[fft_start+peak],1e-6*np.abs(D181022A_trans[fft_start:fft_cutoff])[peak])
            plt.annotate(np.format_float_scientific(1e-4*D181022A_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0),[1.0e-4*D181022A_f_array[fft_start+peak],1.02e-6*np.abs(D181022A_trans[fft_start:fft_cutoff])[peak]])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D181022A')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D181022A_f_array[fft_start:fft_cutoff],np.angle(D181022A_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')





    # Data analysis for D230831B_ref dark, 09.27.2023
    if 1==0:


        file_path = "Z:\\User\\Thomas\\D230831B_ref"
        file_name = "D230831B_ref_SdH1586_dark1K.dat"
        D230831B_ref_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],data_headings=["field","Rxx_x", "Rxy_x", "Rxx_y", "Rxy_y"])
        # D230831B_ref_data.dropna(inplace=True)
        print(D230831B_ref_data)

        # windows: [start_point, chop_point]
        # window = [-1,0] # Full curve
        # window = [-305,3530] # SdH Oscillations
        # window = [-305,4030] # Spin-splitting oscillations
        # window = [-515,3500] # Non-split oscillations and higher-order peaks
        window = [-575,3100]

        D230831B_ref_R_pos , D230831B_ref_B_pos = apodize_data(D230831B_ref_data,["xx"], order=1,start_point=window[0],chop_point = window[1],invert=True, show_plot=True) # Full data window
        D230831B_ref_R_inv , D230831B_ref_B_inv , D230831B_ref_delt_B = interpolate_data(D230831B_ref_R_pos, D230831B_ref_B_pos,invert=True,scaling_order=3)
        # print(len(D230831B_ref_R_pos))
        # D230831B_ref_R_inv = apod_NB(D230831B_ref_R_inv,D230831B_ref_B_inv,order=1,show_plot=True,invert=False)
        # print(D230831B_ref_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(D230831B_ref_data.An_Field,np.abs(D230831B_ref_data.Rxx_x + 1.j *D230831B_ref_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D230831B_ref_data.An_Field,D230831B_ref_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        plt.vlines(D230831B_ref_B_pos[0],-0.1,1.2*np.amax(D230831B_ref_data.Rxx_x),color="r",linestyle="dashed")
        plt.vlines(D230831B_ref_B_pos[-1],-0.1,1.2*np.amax(D230831B_ref_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 1 K, sample D230831B_ref")
        plt.ylim(0,400)
        plt.xlim(-4,0)
        plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/D230831B_ref_B_inv,D230831B_ref_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D230831B_ref_delt_B_inv = 1/D230831B_ref_B_inv[1:-1] - 1/D230831B_ref_B_inv[0:-2]
        D230831B_ref_delt_B_inv_av = np.mean(D230831B_ref_delt_B_inv)

        n_points = 8*len(D230831B_ref_R_inv)
        D230831B_ref_trans = ft.rfft(D230831B_ref_R_inv,n=n_points)
        D230831B_ref_f_array =  np.arange(len(D230831B_ref_trans)) / n_points / np.abs(D230831B_ref_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 300#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_ref_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_ref_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peaks = sig.find_peaks(1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff]), height = 1e-2,prominence=(8e-3,None))
        print(peaks)
        # peak_density = D230831B_ref_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831B_ref_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831B_ref_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*D230831B_ref_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        for peak in peaks[0]:
            plt.scatter(1e-4*D230831B_ref_f_array[fft_start+peak],1e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff])[peak])
            plt.annotate(np.format_float_scientific(1e-4*D230831B_ref_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.05e-4*D230831B_ref_f_array[fft_start+peak],0.9e-6*np.abs(D230831B_ref_trans[fft_start:fft_cutoff])[peak]])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831B_ref')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831B_ref_f_array[fft_start:fft_cutoff],np.angle(D230831B_ref_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')




    # Data analysis for D230831A_b, 09.23.2023
    if 1==0:


        file_path = "Z:\\User\\Thomas\\D230831B_ref"
        file_name = "D230831B_ref_SdH1586_dark1K.dat"
        D230831A_b_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],data_headings=["field","col_1", "col_3", "col_2", "col_4"])
        # D230831A_b_data.dropna(inplace=True)
        print(D230831A_b_data)

        # windows: [start_point, chop_point]
        window = [-1000, 400]

        D230831A_b_R_pos , D230831A_b_B_pos = apodize_data(D230831A_b_data,["xx"], order=1, extra_point_inds=[300],start_point=window[0],chop_point = window[1],invert=False, show_plot=True) # Full data window
        D230831A_b_R_inv , D230831A_b_B_inv , D230831A_b_delt_B = interpolate_data(D230831A_b_R_pos, D230831A_b_B_pos,invert=False,scaling_order=3)
        print(len(D230831A_b_R_pos))
        # D230831A_b_R_inv = apod_NB(D230831A_b_R_inv,D230831A_b_B_inv,order=1,show_plot=True,invert=False)
        # print(D230831A_b_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(D230831A_b_data.An_Field,np.abs(D230831A_b_data.Rxx_x + 1.j *D230831A_b_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        # plt.vlines(D230831A_b_B_pos[0],-0.1,1.2*np.amax(D230831A_b_data.Rxx_x),color="r",linestyle="dashed")
        # plt.vlines(D230831A_b_B_pos[-1],-0.1,1.2*np.amax(D230831A_b_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Illuminated Long. Resistance, $T$ = 1.2 K, sample D230831A_b")
        # plt.ylim(0,0.7e-4)
        # plt.xlim(0,0.8)
        plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/D230831A_b_B_inv,D230831A_b_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D230831A_b_delt_B_inv = 1/D230831A_b_B_inv[1:-1] - 1/D230831A_b_B_inv[0:-2]
        D230831A_b_delt_B_inv_av = np.mean(D230831A_b_delt_B_inv)

        n_points = 8*len(D230831A_b_R_inv)
        D230831A_b_trans = ft.rfft(D230831A_b_R_inv,n=n_points)
        D230831A_b_f_array =  np.arange(len(D230831A_b_trans)) / n_points / np.abs(D230831A_b_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 400#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831A_b_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831A_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831A_b_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831A_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peak_density = D230831A_b_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831A_b_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831A_b_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*D230831A_b_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831A_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        plt.scatter(1e-4*peak_density,1e-6*np.amax(np.abs(D230831A_b_trans[fft_start:fft_cutoff])))
        plt.annotate(r"peak = " + np.format_float_scientific(1e-4*peak_density, unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.1e-4*peak_density, 0.9e-6*np.amax(np.abs(D230831A_b_trans[fft_start:fft_cutoff]))])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831A_b')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831A_b_f_array[fft_start:fft_cutoff],np.angle(D230831A_b_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')





    # Data analysis for D230831B_b, 09.25.2023
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\ETH Zurich Materials\\samples\\D230831B_b\\"
        file_name = "D230831B_b_magnetotransport_ill1K2.txt"
        D230831B_b_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],data_headings=["field","col_1", "col_3", "col_2", "col_4"])
        # D230831B_b_data.dropna(inplace=True)
        print(D230831B_b_data)

        # windows: [start_point, chop_point]
        window = [-1000, 1150]

        D230831B_b_R_pos , D230831B_b_B_pos = apodize_data(D230831B_b_data,["xx"], order=1, extra_point_inds=[300],start_point=window[0],chop_point = window[1],invert=False, show_plot=True) # Full data window
        D230831B_b_R_inv , D230831B_b_B_inv , D230831B_b_delt_B = interpolate_data(D230831B_b_R_pos, D230831B_b_B_pos,invert=False,scaling_order=3)
        print(len(D230831B_b_R_pos))
        # D230831B_b_R_inv = apod_NB(D230831B_b_R_inv,D230831B_b_B_inv,order=1,show_plot=True,invert=False)
        # print(D230831B_b_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(D230831B_b_data.An_Field,np.abs(D230831B_b_data.Rxx_x + 1.j *D230831B_b_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D230831B_b_data.An_Field,D230831B_b_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D230831B_b_data.An_Field,D230831B_b_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D230831B_b_data.An_Field,D230831B_b_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D230831B_b_data.An_Field,D230831B_b_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        plt.vlines(D230831B_b_B_pos[0],-0.1,1.2*np.amax(D230831B_b_data.Rxx_x),color="r",linestyle="dashed")
        plt.vlines(D230831B_b_B_pos[-1],-0.1,1.2*np.amax(D230831B_b_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 2.5 K, sample D230831A")
        # plt.ylim(0,0.7e-4)
        # plt.xlim(0,0.8)
        # plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/D230831B_b_B_inv,D230831B_b_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D230831B_b_delt_B_inv = 1/D230831B_b_B_inv[1:-1] - 1/D230831B_b_B_inv[0:-2]
        D230831B_b_delt_B_inv_av = np.mean(D230831B_b_delt_B_inv)

        n_points = 8*len(D230831B_b_R_inv)
        D230831B_b_trans = ft.rfft(D230831B_b_R_inv,n=n_points)
        D230831B_b_f_array =  np.arange(len(D230831B_b_trans)) / n_points / np.abs(D230831B_b_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 150#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831B_b_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831B_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831B_b_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831B_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peaks = sig.find_peaks(1e-6*np.abs(D230831B_b_trans[fft_start:fft_cutoff]), height = 6e-4)
        print(peaks)
        # peak_density = D230831B_b_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831B_b_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831B_b_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*D230831B_b_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831B_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        for peak in peaks[0]:
            plt.scatter(1e-4*D230831B_b_f_array[fft_start+peak],1e-6*np.abs(D230831B_b_trans[fft_start:fft_cutoff])[peak])
            plt.annotate(np.format_float_scientific(1e-4*D230831B_b_f_array[fft_start+peak], unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.05e-4*D230831B_b_f_array[fft_start+peak],0.9e-6*np.abs(D230831B_b_trans[fft_start:fft_cutoff])[peak]])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831B_b')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831B_b_f_array[fft_start:fft_cutoff],np.angle(D230831B_b_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')


# Data analysis for D230831A_b, 09.23.2023
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\ETH Zurich Materials\\samples\\D230831B_b"
        file_name = "D230831A_b_magnetotransport_ill1.3K_data.txt"
        D230831A_b_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],data_headings=["field","col_1", "col_3", "col_2", "col_4"])
        # D230831A_b_data.dropna(inplace=True)
        print(D230831A_b_data)

        # windows: [start_point, chop_point]
        window = [-1000, 400]

        D230831A_b_R_pos , D230831A_b_B_pos = apodize_data(D230831A_b_data,["xx"], order=1, extra_point_inds=[300],start_point=window[0],chop_point = window[1],invert=False, show_plot=True) # Full data window
        D230831A_b_R_inv , D230831A_b_B_inv , D230831A_b_delt_B = interpolate_data(D230831A_b_R_pos, D230831A_b_B_pos,invert=False,scaling_order=3)
        print(len(D230831A_b_R_pos))
        # D230831A_b_R_inv = apod_NB(D230831A_b_R_inv,D230831A_b_B_inv,order=1,show_plot=True,invert=False)
        # print(D230831A_b_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(D230831A_b_data.An_Field,np.abs(D230831A_b_data.Rxx_x + 1.j *D230831A_b_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D230831A_b_data.An_Field,D230831A_b_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        # plt.vlines(D230831A_b_B_pos[0],-0.1,1.2*np.amax(D230831A_b_data.Rxx_x),color="r",linestyle="dashed")
        # plt.vlines(D230831A_b_B_pos[-1],-0.1,1.2*np.amax(D230831A_b_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Illuminated Long. Resistance, $T$ = 1.2 K, sample D230831A_b")
        # plt.ylim(0,0.7e-4)
        # plt.xlim(0,0.8)
        plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/D230831A_b_B_inv,D230831A_b_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D230831A_b_delt_B_inv = 1/D230831A_b_B_inv[1:-1] - 1/D230831A_b_B_inv[0:-2]
        D230831A_b_delt_B_inv_av = np.mean(D230831A_b_delt_B_inv)

        n_points = 8*len(D230831A_b_R_inv)
        D230831A_b_trans = ft.rfft(D230831A_b_R_inv,n=n_points)
        D230831A_b_f_array =  np.arange(len(D230831A_b_trans)) / n_points / np.abs(D230831A_b_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 400#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831A_b_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831A_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831A_b_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831A_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peak_density = D230831A_b_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831A_b_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831A_b_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*D230831A_b_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831A_b_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        plt.scatter(1e-4*peak_density,1e-6*np.amax(np.abs(D230831A_b_trans[fft_start:fft_cutoff])))
        plt.annotate(r"peak = " + np.format_float_scientific(1e-4*peak_density, unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.1e-4*peak_density, 0.9e-6*np.amax(np.abs(D230831A_b_trans[fft_start:fft_cutoff]))])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831A_b')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831A_b_f_array[fft_start:fft_cutoff],np.angle(D230831A_b_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')



    # Data analysis for D230831A_a, 09.23.2023
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\ETH Zurich Materials\\samples\\D230831A_a\\"
        file_name = "D230831A_a_2.5K_magnetotransport.txt"
        D230831A_a_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],data_headings=["field","col_1", "col_3", "col_2", "col_4"])
        # D230831A_a_data.dropna(inplace=True)
        print(D230831A_a_data)

        # windows: [start_point, chop_point]
        window = [-1000, 400]

        D230831A_a_R_pos , D230831A_a_B_pos = apodize_data(D230831A_a_data,["xx"], order=1, extra_point_inds=[300],start_point=window[0],chop_point = window[1],invert=True, show_plot=True) # Full data window
        D230831A_a_R_inv , D230831A_a_B_inv , D230831A_a_delt_B = interpolate_data(D230831A_a_R_pos, D230831A_a_B_pos,invert=True,scaling_order=3)
        print(len(D230831A_a_R_pos))
        # D230831A_a_R_inv = apod_NB(D230831A_a_R_inv,D230831A_a_B_inv,order=1,show_plot=True,invert=False)
        # print(D230831A_a_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(D230831A_a_data.An_Field,np.abs(D230831A_a_data.Rxx_x + 1.j *D230831A_a_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(D230831A_a_data.An_Field,D230831A_a_data.Rxx_x ,color="b",label=r"Rxx_x")
        # plt.plot(D230831A_a_data.An_Field,D230831A_a_data.Rxy_x,color="r",label=r"Rxy_x")
        # plt.plot(D230831A_a_data.An_Field,D230831A_a_data.Rxx_y,color="b",linestyle="dashed",label=r"Rxx_y")
        # plt.plot(D230831A_a_data.An_Field,D230831A_a_data.Rxy_y,color="r",linestyle="dashed",label=r"Rxy_y")
        plt.vlines(D230831A_a_B_pos[0],-0.1,1.2*np.amax(D230831A_a_data.Rxx_x),color="r",linestyle="dashed")
        plt.vlines(D230831A_a_B_pos[-1],-0.1,1.2*np.amax(D230831A_a_data.Rxx_x),color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 2.5 K, sample D230831A")
        # plt.ylim(0,0.7e-4)
        # plt.xlim(0,0.8)
        # plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/D230831A_a_B_inv,D230831A_a_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        D230831A_a_delt_B_inv = 1/D230831A_a_B_inv[1:-1] - 1/D230831A_a_B_inv[0:-2]
        D230831A_a_delt_B_inv_av = np.mean(D230831A_a_delt_B_inv)

        n_points = 8*len(D230831A_a_R_inv)
        D230831A_a_trans = ft.rfft(D230831A_a_R_inv,n=n_points)
        D230831A_a_f_array =  np.arange(len(D230831A_a_trans)) / n_points / np.abs(D230831A_a_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 400#-3520
        # # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(D230831A_a_f_array[fft_start:fft_cutoff],1e-6*np.real(D230831A_a_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(D230831A_a_f_array[fft_start:fft_cutoff],1e-6*np.imag(D230831A_a_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peak_density = D230831A_a_f_array[fft_start:fft_cutoff][indexOf(np.abs(D230831A_a_trans[fft_start:fft_cutoff]),np.amax(np.abs(D230831A_a_trans[fft_start:fft_cutoff])))]
        # print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*D230831A_a_f_array[fft_start:fft_cutoff],1e-6*np.abs(D230831A_a_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        plt.scatter(1e-4*peak_density,1e-6*np.amax(np.abs(D230831A_a_trans[fft_start:fft_cutoff])))
        plt.annotate(r"peak = " + np.format_float_scientific(1e-4*peak_density, unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.1e-4*peak_density, 0.9e-6*np.amax(np.abs(D230831A_a_trans[fft_start:fft_cutoff]))])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D230831A_a')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*D230831A_a_f_array[fft_start:fft_cutoff],np.angle(D230831A_a_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')


    # 09/18/2023 ETH sample D210611A (Loki) Data Anlysis
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\GraysonGroupResearch\\Degeneracy Cooling Project\\QHE - ETH\\Data"
        file_name = "D210611A_Loki.txt"
        Loki_data = get_dat_data(file_path,file_name,R_ind = ["ETH"],data_headings=["field","Uxx (X)", "Uxx (Y)", "Uxy (X)", "Uxy (Y)"],has_header=True)
        # Loki_data.dropna(inplace=True)
        print(Loki_data)

        # windows: [start_point, chop_point]
        # window = [-3000, 600]
        # window = [-4600,600] # Bad , 10 - 16
        # window = [-5200, 150] # Eh
        # window = [-4210, 1072] # Good one! , 2.5-5, use scaling_order = 0.8
        # window = [-4420 , 1000] # Decent? 5-10
        # window = [-4100, 1072]
        # window = [-4120, 1065] # Even better, 2-6, use scaling_order = 1.0
        # window = [-4000, 1065] # Even better, 2-6, use scaling_order = 1.0
        window = [-3000,1565]

        Loki_R_pos , Loki_B_pos = apodize_data(Loki_data,["xx"], order=1, extra_point_inds=[300],start_point=window[0],chop_point = window[1],invert=False, show_plot=True) # Full data window
        Loki_R_inv , Loki_B_inv = interpolate_data(Loki_R_pos, Loki_B_pos,invert=False,scaling_order=1)
        print(len(Loki_R_pos))
        # Loki_R_inv = apod_NB(Loki_R_inv,Loki_B_inv,order=1,show_plot=True,invert=False)
        # print(Loki_delt_B)

        # Plot raw data
        plt.figure()
        # plt.plot(Loki_data.An_Field,np.abs(Loki_data.Rxx_x + 1.j *Loki_data.Rxx_y) ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        plt.plot(Loki_data.An_Field,Loki_data.Rxx_x ,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(Loki_data.An_Field,Loki_data.Ryy_x,color="b",label=r"Re{$R_\mathrm{yy}$}, single lock-in")
        # plt.plot(Loki_data.An_Field,Loki_data.Rxx_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(Loki_data.An_Field,Loki_data.Ryy_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{yy}$}, single lock-in")
        plt.vlines(Loki_B_pos[0],-0.1,1.5e-4,color="r",linestyle="dashed")
        plt.vlines(Loki_B_pos[-1],-0.1,1.5e-4,color="r",linestyle="dashed")
        plt.ylabel(r"$R_{\rm xx}$ ($\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 25 mK, sample D210611A")
        plt.ylim(0,1.7e-4)
        # plt.xlim(0,0.8)
        # plt.legend()


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/Loki_B_inv,Loki_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, after background subtraction and scaling")


        Loki_delt_B_inv = 1/Loki_B_inv[1:-1] - 1/Loki_B_inv[0:-2]
        Loki_delt_B_inv_av = np.mean(Loki_delt_B_inv)

        n_points = 8*len(Loki_R_inv)
        Loki_trans = ft.rfft(Loki_R_inv,n=n_points)
        Loki_f_array =  np.arange(len(Loki_trans)) / n_points / np.abs(Loki_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 1000#-3520
        # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(Loki_f_array[fft_start:fft_cutoff],1e-6*np.real(Loki_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(Loki_f_array[fft_start:fft_cutoff],1e-6*np.imag(Loki_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')

        plt.figure()
        # plt.subplots(2,1)
        
        peak_density = Loki_f_array[fft_start:fft_cutoff][indexOf(np.abs(Loki_trans[fft_start:fft_cutoff]),np.amax(np.abs(Loki_trans[fft_start:fft_cutoff])))]
        print("Density n =  ",peak_density*1e-4,r" cm^-2$")
        # plt.subplot(2,1,1)
        plt.plot(1e-4*Loki_f_array[fft_start:fft_cutoff],1e-6*np.abs(Loki_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        plt.scatter(1e-4*peak_density,1e-6*np.amax(np.abs(Loki_trans[fft_start:fft_cutoff])))
        plt.annotate(r"$n$ = " + np.format_float_scientific(1e-4*peak_density, unique = False, precision=2,exp_digits=0)+ r" cm$^{-2}$",[1.1e-4*peak_density, 0.9e-6*np.amax(np.abs(Loki_trans[fft_start:fft_cutoff]))])
        plt.ylabel(r'FFT Amplitude')
        plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        plt.title(r'FFT in 1/B of Processed Long. Resistance, sample D210611A')

        # plt.subplot(2,1,2)
        # plt.plot(1e-4*Loki_f_array[fft_start:fft_cutoff],np.angle(Loki_trans[fft_start:fft_cutoff]))
        # # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.xlabel(r"$n_\mathrm{2D}$ (cm$^{-2}$)")
        # plt.ylabel(r'Phase (rads)')







        # 08/02/2023 ETH Data Anlysis
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\GraysonGroupResearch\\Degeneracy Cooling Project\\QHE - ETH\\Data"
        file_name = "D230803A_SdH1524_dark1K.dat"
        ETH_data = get_dat_data(file_path,file_name,R_ind = ["ETH"])
        # ETH_data.dropna(inplace=True)
        print(ETH_data)


        # Plot raw data
        plt.figure()
        plt.plot(ETH_data.An_Field,ETH_data.Rxx_x,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(ETH_data.An_Field,ETH_data.Ryy_x,color="b",label=r"Re{$R_\mathrm{yy}$}, single lock-in")
        # plt.plot(ETH_data.An_Field,ETH_data.Rxx_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(ETH_data.An_Field,ETH_data.Ryy_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{yy}$}, single lock-in")
        plt.ylabel(r"$R_{\rm Hall}$ (k$\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 1.8 K, 8/2/23 ")
        # plt.ylim(0,700)
        plt.legend()

        ETH_R_pos , ETH_B_pos = apodize_data(ETH_data,["xx"], order=1, extra_point_inds=[300],start_point=-240,chop_point = 200,invert=True, show_plot=True) # Full data window
        ETH_R_inv , ETH_B_inv , ETH_delt_B = interpolate_data(ETH_R_pos, ETH_B_pos,invert=True,scaling_order=3)
        ETH_R_inv = apod_NB(ETH_R_inv,ETH_B_inv,order=1,show_plot=True,invert=True)
        # print(ETH_delt_B)


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/ETH_B_inv,ETH_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, with apodization")


        ETH_delt_B_inv = 1/ETH_B_inv[1:-1] - 1/ETH_B_inv[0:-2]
        ETH_delt_B_inv_av = np.mean(ETH_delt_B_inv)
        # # Plot interval of 1/B between datapoints
        # plt.figure()
        # plt.plot(m9_p9_delt_B_inv)
        # print(m9_p9_delt_B_inv_av)

        n_points = 8*len(ETH_R_inv)
        ETH_trans = ft.rfft(ETH_R_inv,n=n_points)
        ETH_f_array =  np.arange(len(ETH_trans)) / n_points / np.abs(ETH_delt_B_inv_av) *c.e / c.h


        # Plot FFT's
        fft_cutoff = 300
        plt.subplots(2,1)

        plt.subplot(2,1,1)
        plt.plot(ETH_f_array[2:fft_cutoff],1e-6*np.real(ETH_trans[2:fft_cutoff]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Real Amplitude')
        plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        plt.subplot(2,1,2)
        plt.plot(ETH_f_array[2:fft_cutoff],1e-6*np.imag(ETH_trans[2:fft_cutoff]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Amplitude')


        plt.subplots(2,1)

        plt.subplot(2,1,1)
        plt.plot(ETH_f_array[2:fft_cutoff],1e-6*np.abs(ETH_trans[2:fft_cutoff]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Imaginary Amplitude')
        plt.title(r'Polar FFT in 1/B of Long. Resistance, with apodization')

        plt.subplot(2,1,2)
        plt.plot(ETH_f_array[2:200],np.angle(ETH_trans[2:200]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Phase (rads)')







    # 08/02/2023 Data 0.1 T to -8 T R_yy Fourier analysis
    if 1==0:
        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\GraysonGroupResearch\\Degeneracy Cooling Project\\QHE - 08022023\\Data"
        file_name = "230802_057_GaAs_D181211Ai_1uA_LI110_Vyy_SS_1K8_Bsweep_p0T1tom8T.csv"
        p01_m8_data = get_csv_data(file_path,file_name,R_ind = ["yy"])
        p01_m8_data.dropna(inplace=True)
        # print(p01_m8_data.Ryy_x)


        # # Plot raw data
        # plt.figure()
        # # plt.plot(p01_m8_data.An_Field,p01_m8_data.Rxx_x,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(p01_m8_data.An_Field,p01_m8_data.Ryy_x,color="b",label=r"Re{$R_\mathrm{yy}$}, single lock-in")
        # # plt.plot(p01_m8_data.An_Field,p01_m8_data.Rxx_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{xx}$}, single lock-in")
        # # plt.plot(p01_m8_data.An_Field,p01_m8_data.Ryy_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{yy}$}, single lock-in")
        # plt.ylabel(r"$R_{\rm Hall}$ (k$\mathrm{\Omega{}}$)")
        # plt.xlabel(r"$B$ (T)")
        # plt.title(r"Long. Resistance, $T$ = 1.8 K, 8/2/23 ")
        # plt.ylim(0,700)
        # plt.legend()

        p01_m8_R_pos , p01_m8_B_pos = apodize_data(p01_m8_data,["yy"], order=2, extra_point_inds=[-80],chop_point = 80,invert=True, show_plot=True)
        p01_m8_R_inv , p01_m8_B_inv , p01_m8_delt_B = interpolate_data(p01_m8_R_pos, p01_m8_B_pos)


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/p01_m8_B_inv,p01_m8_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, with apodization")


        p01_m8_trans = ft.rfft(p01_m8_R_inv)
        p01_m8_f_array =  np.arange(len(p01_m8_trans)+1) / p01_m8_delt_B


        # Plot FFT's
        fft_cutoff = 100
        plt.subplots(2,1)

        plt.subplot(2,1,1)
        plt.plot(p01_m8_f_array[2:fft_cutoff],1e-6*np.real(p01_m8_trans[2:fft_cutoff]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Real Amplitude')
        plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        plt.subplot(2,1,2)
        plt.plot(p01_m8_f_array[2:fft_cutoff],1e-6*np.imag(p01_m8_trans[2:fft_cutoff]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Amplitude')


        plt.subplots(2,1)

        plt.subplot(2,1,1)
        plt.plot(p01_m8_f_array[2:fft_cutoff],1e-6*np.abs(p01_m8_trans[2:fft_cutoff]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Imaginary Amplitude')
        plt.title(r'Polar FFT in 1/B of Long. Resistance, with apodization')

        plt.subplot(2,1,2)
        plt.plot(p01_m8_f_array[2:200],np.angle(p01_m8_trans[2:200]))
        plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Phase (rads)')





    # 08/02/2023 Data -8 T to 9 T R_xx Fourier analysis
    if 1==0:


        file_path = "C:\\Users\\thoma\\OneDrive\\Documents\\Research Materials\\GraysonGroupResearch\\Degeneracy Cooling Project\\QHE - 08022023\\Data"
        file_name = "230802_059_GaAs_D181211Ai_1uA_LI110_pVyy_L111_oVxx_L112_Vxx_1K8_Bsweep_m9Ttop9T.csv"
        m9_p9_data = get_csv_data(file_path,file_name,R_ind = ["xx"])
        m9_p9_data.dropna(inplace=True)
        # print(m9_p9_data.Ryy_x)


        # Plot raw data
        plt.figure()
        plt.plot(m9_p9_data.An_Field,m9_p9_data.Rxx_x,color="b",label=r"Re{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(m9_p9_data.An_Field,m9_p9_data.Ryy_x,color="b",label=r"Re{$R_\mathrm{yy}$}, single lock-in")
        # plt.plot(m9_p9_data.An_Field,m9_p9_data.Rxx_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{xx}$}, single lock-in")
        # plt.plot(m9_p9_data.An_Field,m9_p9_data.Ryy_y,color="b",linestyle="dashed",label=r"Im{$R_\mathrm{yy}$}, single lock-in")
        plt.ylabel(r"$R_{\rm Hall}$ (k$\mathrm{\Omega{}}$)")
        plt.xlabel(r"$B$ (T)")
        plt.title(r"Long. Resistance, $T$ = 1.8 K, 8/2/23 ")
        # plt.ylim(0,700)
        plt.legend()


        m9_p9_R_pos , m9_p9_B_pos = apodize_data(m9_p9_data,["xx"], order=0, extra_point_inds=[300],start_point=-70,chop_point = 0,invert=True, show_plot=True) # Full data window
        # m9_p9_R_pos , m9_p9_B_pos = apodize_data(m9_p9_data,["xx"], order=0, extra_point_inds=[300],start_point=-70,chop_point = 1350,invert=True, show_plot=True) # Low-B data window
        # m9_p9_R_pos , m9_p9_B_pos = apodize_data(m9_p9_data,["xx"], order=0, extra_point_inds=[300],start_point=-150,chop_point = 0, invert=True, show_plot=True) # High-B Data Window
        m9_p9_R_inv , m9_p9_B_inv , m9_p9_delt_B = interpolate_data(m9_p9_R_pos, m9_p9_B_pos,invert=True)

        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/m9_p9_B_inv,m9_p9_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, without apodization")

        m9_p9_R_inv = apod_NB(m9_p9_R_inv,m9_p9_B_inv,order=3,show_plot=True,invert=True)


        # Plot Inverted Data after apodization and interpolation
        plt.figure()
        plt.plot(1/m9_p9_B_inv,m9_p9_R_inv)
        plt.xlabel(r"$1/B$ (T$^{-1}$)")
        plt.ylabel(r"$R_\mathrm{xx}$ ($\Omega$)")
        plt.title(r"Long. Resistance vs 1/B, with apodization")

        m9_p9_delt_B_inv = 1/m9_p9_B_inv[1:-1] - 1/m9_p9_B_inv[0:-2]
        m9_p9_delt_B_inv_av = np.mean(m9_p9_delt_B_inv)
        # # Plot interval of 1/B between datapoints
        # plt.figure()
        # plt.plot(m9_p9_delt_B_inv)
        # print(m9_p9_delt_B_inv_av)

        n_points = 8*len(m9_p9_R_inv)
        m9_p9_trans = ft.rfft(m9_p9_R_inv,n=n_points)
        m9_p9_f_array =  np.arange(len(m9_p9_trans)) / n_points / np.abs(m9_p9_delt_B_inv_av) *c.e / c.h

        # Plot FFT's
        fft_start = 0#3520
        fft_cutoff = 200#-3520
        # plt.subplots(2,1)

        # plt.subplot(2,1,1)
        # plt.plot(m9_p9_f_array[fft_start:fft_cutoff],1e-6*np.real(m9_p9_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Real Amplitude')
        # plt.title(r'Cartesian FFT in 1/B of Long. Resistance, with apodization')

        # plt.subplot(2,1,2)
        # plt.plot(m9_p9_f_array[fft_start:fft_cutoff],1e-6*np.imag(m9_p9_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        # plt.ylabel(r'Imaginary Amplitude')


        plt.subplots(2,1)

        plt.subplot(2,1,1)
        plt.plot(m9_p9_f_array[fft_start:fft_cutoff],1e-6*np.abs(m9_p9_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        plt.ylabel(r'Amplitude')
        plt.title(r'Polar FFT in 1/B of Long. Resistance, with apodization')

        plt.subplot(2,1,2)
        plt.plot(m9_p9_f_array[fft_start:fft_cutoff],np.angle(m9_p9_trans[fft_start:fft_cutoff]))
        # plt.xlabel(r"$f_{1/B}$ (T)")
        plt.xlabel(r"$n_\mathrm{2D}$ (m$^{-2}$)")
        plt.ylabel(r'Phase (rads)')











    plt.show()
