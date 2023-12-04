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









if __name__ == "__main__":

    # D230831B Second Cooldown 450 mV data
    if 1==0:
        file_path = "z:\\User\\Thomas\\D230831B 2nd cooldown\\02 full sweeps"
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
        nu_bounds.append((1443,1660)) # nu = 1
        nu_bounds.append((980,1097)) # nu = 2
        nu_bounds.append((735,802)) # nu = 3
        nu_bounds.append((590,650)) # nu = 4
        nu_bounds.append((525,549)) # nu = 5
        nu_bounds.append((465,484)) # nu = 6
        nu_bounds.append((437,440)) # nu = 7
        # nu_bounds.append((0,0)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        # plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        # plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        # plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        # plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        # plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        # plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        # plt.xlim(0,4)
        plt.title(r"Long. Resistivity with minima start & endpoints, $V_\mathrm{g}$ ="+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()

        plt.figure()
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
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()




    # D230831B Second Cooldown 150 mV data
    if 1==0:
        file_path = "z:\\User\\Thomas\\D230831B 2nd cooldown\\02 full sweeps"
        Vg = 150
        file_name = "D230831B_" + np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-").replace(" ","0") + "mV_Vg.dat"
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
        nu_bounds.append((1448,1720)) # nu = 1
        nu_bounds.append((995,1095)) # nu = 2
        nu_bounds.append((753,832)) # nu = 3
        nu_bounds.append((597,655)) # nu = 4
        nu_bounds.append((540,549)) # nu = 5
        nu_bounds.append((470,485)) # nu = 6
        # nu_bounds.append((437,440)) # nu = 7
        # nu_bounds.append((0,0)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        # plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        plt.ylim(-0.00001,0.005)
        plt.xlim(0,4)
        plt.title(r"Long. Resistivity with minima start & endpoints, $V_\mathrm{g}$ ="+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()

        plt.figure()
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
        # nu=7
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu7[nu_bounds[nu][0]:nu_bounds[nu][1]],color='cyan',ls=":")
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()



    # D230831B Second Cooldown 100 mV data
    if 1==0:
        file_path = "z:\\User\\Thomas\\D230831B 2nd cooldown\\02 full sweeps"
        Vg = 100#[-200, -100, 0]
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
        nu_bounds.append((1443,1660)) # nu = 1
        nu_bounds.append((980,1097)) # nu = 2
        nu_bounds.append((735,802)) # nu = 3
        nu_bounds.append((590,650)) # nu = 4
        nu_bounds.append((525,549)) # nu = 5
        nu_bounds.append((465,484)) # nu = 6
        nu_bounds.append((437,440)) # nu = 7
        # nu_bounds.append((0,0)) # nu = 8

        # plt.plot(inv.Rxx[nu_bounds[3][0]:nu_bounds[3][1]])

        plt.figure()
        # plt.plot(inv.Rxx[0:500])
        # plt.plot(inv.Rxx)
        plt.plot(inv.B_field,inv.Rxx)
        plt.scatter([inv.B_field[nu_bounds[1][0]],inv.B_field[nu_bounds[1][1]]],[inv.Rxx[nu_bounds[1][0]],inv.Rxx[nu_bounds[1][1]]],color="b",label=r"$\nu$= 1")
        plt.scatter([inv.B_field[nu_bounds[2][0]],inv.B_field[nu_bounds[2][1]]],[inv.Rxx[nu_bounds[2][0]],inv.Rxx[nu_bounds[2][1]]],color="r",label=r"$\nu$= 2")
        plt.scatter([inv.B_field[nu_bounds[3][0]],inv.B_field[nu_bounds[3][1]]],[inv.Rxx[nu_bounds[3][0]],inv.Rxx[nu_bounds[3][1]]],color="orange",label=r"$\nu$= 3")
        plt.scatter([inv.B_field[nu_bounds[4][0]],inv.B_field[nu_bounds[4][1]]],[inv.Rxx[nu_bounds[4][0]],inv.Rxx[nu_bounds[4][1]]],color="purple",label=r"$\nu$= 4")
        plt.scatter([inv.B_field[nu_bounds[5][0]],inv.B_field[nu_bounds[5][1]]],[inv.Rxx[nu_bounds[5][0]],inv.Rxx[nu_bounds[5][1]]],color="yellow",label=r"$\nu$= 5")
        plt.scatter([inv.B_field[nu_bounds[6][0]],inv.B_field[nu_bounds[6][1]]],[inv.Rxx[nu_bounds[6][0]],inv.Rxx[nu_bounds[6][1]]],color="pink",label=r"$\nu$= 6")
        plt.scatter([inv.B_field[nu_bounds[7][0]],inv.B_field[nu_bounds[7][1]]],[inv.Rxx[nu_bounds[7][0]],inv.Rxx[nu_bounds[7][1]]],color="cyan",label=r"$\nu$= 7")
        # plt.scatter([inv.B_field[nu_bounds[8][0]],inv.B_field[nu_bounds[8][1]]],[inv.Rxx[nu_bounds[8][0]],inv.Rxx[nu_bounds[8][1]]],color="green",label=r"$\nu$= 8")
        plt.grid()
        # plt.xlim(0,4)
        plt.title(r"Long. Resistivity with minima start & endpoints, $V_\mathrm{g}$ ="+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.ylim(-0.00001,0.005)
        plt.xlim(0,4)
        plt.legend()

        plt.figure()
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
        # nu=8
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.p_xx_tot[nu_bounds[nu][0]:nu_bounds[nu][1]],color='k')
        # plt.plot(inv.B_field[nu_bounds[nu][0]:nu_bounds[nu][1]],inv.rho_xx_par_nu8[nu_bounds[nu][0]:nu_bounds[nu][1]],color='green',ls=":")
        plt.title(r"Parallel Resistivity, $\nu$ = 1-"+str(nu))
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()





    # D230831B Second Cooldown 000 mV data
    if 1==0:
        file_path = "z:\\User\\Thomas\\D230831B 2nd cooldown\\02 full sweeps"
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
        plt.title(r"Long. Resistivity with minima start & endpoints, $V_\mathrm{g}$ ="+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")
        plt.ylabel(r"$\rho_\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()
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
        plt.ylabel(r"$\rho\mathrm{xx}$ ($\Omega$)")
        plt.xlabel(r"$B$ (T)")
        plt.legend()







    # D230831B Second Cooldown Combined Plots
    if 1==1:
        file_path = "z:\\User\\Thomas\\D230831B 2nd cooldown\\02 full sweeps"
        Vg_vals = [000, 100, 150,450, 500]

        for Vg in Vg_vals:
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
            nu_bounds.append((1443,1660)) # nu = 1
            nu_bounds.append((980,1097)) # nu = 2
            nu_bounds.append((735,802)) # nu = 3
            nu_bounds.append((590,650)) # nu = 4
            nu_bounds.append((525,549)) # nu = 5
            nu_bounds.append((465,484)) # nu = 6
            nu_bounds.append((437,440)) # nu = 7
            # nu_bounds.append((0,0)) # nu = 8


            plt.figure(num=1)
            ax1 = plt.gca()
            ax1.plot(inv.B_field,1e3*inv.p_xx_tot,label=r"$V_\mathrm{g}$= "+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")

            plt.figure(num=2)
            ax2 = plt.gca()
            ax2.plot(inv.B_field,1e3*inv.p_xx_tot,label=r"$V_\mathrm{g}$= "+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")

            plt.figure(num=3)
            ax3 = plt.gca()
            ax3.plot(inv.B_field,1e3*inv.p_xx_tot,label=r"$V_\mathrm{g}$= "+np.format_float_positional(Vg,unique=False,pad_left=3,precision=3,trim="-")+"mV")



            

        plt.figure(num=1)
        ax1.set_title(r"Long. Resistivity for different $V_\mathrm{g}$, Sample D230831B, $T$= 15 mK")
        ax1.set_ylabel(r"$\rho_\mathrm{xx}$ (m$\Omega$)")
        ax1.set_xlabel(r"$B$ (T)")
        ax1.set_xlim(0,1)
        ax1.set_ylim(bottom=-0.01,top=0.2)
        ax1.grid()
        ax1.legend()

        plt.figure(num=1)
        ax2.set_title(r"Long. Resistivity for different $V_\mathrm{g}$, Sample D230831B, $T$= 15 mK")
        ax2.set_ylabel(r"$\rho_\mathrm{xx}$ (m$\Omega$)")
        ax2.set_xlabel(r"$B$ (T)")
        ax2.set_xlim(1,5)
        ax2.set_ylim(bottom=-0.01,top=1.3)
        ax2.grid()
        ax2.legend()

        plt.figure(num=1)
        ax3.set_title(r"Long. Resistivity for different $V_\mathrm{g}$, Sample D230831B, $T$= 15 mK")
        ax3.set_ylabel(r"$\rho_\mathrm{xx}$ (m$\Omega$)")
        ax3.set_xlabel(r"$B$ (T)")
        ax3.set_xlim(5,12)
        ax3.grid()
        ax3.legend()






    
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









plt.show()