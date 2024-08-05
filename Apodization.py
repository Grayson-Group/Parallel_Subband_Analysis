# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:55:20 2024

@author: Madma
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import pandas 
import matplotlib.dates as mdates


def apod_NB(y,x,order=0,show_plot=False,invert=False):

    
    #Note: Any instance of R_inv or B_inv is just y or x respectively


    if order == 0:
        return y

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


    fun_NB_1 = lambda z: C0*(1-z**2)**0 + C1*(1-z**2)**1 + C2*(1-z**2)**2 + C4*(1-z**2)**4
    if invert==False:
        norm_x = (2*1/x - 2*np.amin(1/x))/(np.amax(1/x)-np.amin(1/x)) - 1
        apod_fun = fun_NB_1(norm_x)

    if invert==True:
        norm_x = (2*1/x - 2*np.amin(1/x))/(np.amax(1/x)-np.amin(1/x)) - 1
        # norm_B_inv = (1/B_inv - np.amin(1/B_inv))/(np.amax(1/B_inv)-np.amin(1/B_inv))
        # apod_fun = -1*fun_NB_1(norm_B_inv) + np.amin(fun_NB_1(norm_B_inv)) + 1
        apod_fun = fun_NB_1(norm_x)
        # apod_fun = fun_NB_1(norm_B_inv)

    # print("Norm B range: ", norm_B_inv[0], norm_B_inv[-1])

    if show_plot:
        # plt.figure()
        # plt.plot(1/B_inv,R_inv,color="b",label=r"$R_\mathrm{Hall}$")
        # plt.title("")
        # plt.scatter(B_pos[extra_point_inds[0]],R_pos[extra_point_inds[0]])
        # plt.legend()

        plt.figure()
        plt.scatter(1/x,np.amax(y)*apod_fun,color="r",label=label_str)
        plt.plot(1/x, y,color="b",label=r"$R_\mathrm{Long}$")
        plt.legend()
        plt.title(r"Norton-Beer Apodization Function and Resistance Data")

        # plt.figure()
        # # plt.plot(1/B_inv,apod_fun,color="r",label=r"NB1  Apodizing Function")
        # plt.plot(1/B_inv,R_inv * apod_fun,color="b",label=r"$R_\mathrm{Long}$")
        # plt.legend()
        # plt.title(r"Resistance After Norton-Beer Apodization")



    return y * apod_fun
