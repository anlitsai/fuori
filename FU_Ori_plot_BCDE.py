#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:15:21 2020

@author: altsai
"""


#import os
#import sys
#import shutil
import numpy as np
#import csv
#import time
#import math
import pandas as pd
import matplotlib.pyplot as plt

nn=30

obj_name='FU Orion'
filter_ID=['B','V','R']
list_data=['./XPM/BCDE_slt_InstMag_Bmag/Bmag_FU_Ori_all.txt','./XPM/BCDE_slt_InstMag_Vmag/Vmag_FU_Ori_all.txt','./XPM/BCDE_slt_InstMag_Rmag/Rmag_FU_Ori_all.txt']
filter_mag=['Bmag','Vmag','Rmag']
filter_err=['ErrorBmag','ErrorVmag','ErrorRmag']

n_data=len(list_data)

fig,axs=plt.subplots(3,1,figsize=(6,8))
fig.subplots_adjust(hspace=0.5,wspace=0.5)
axs=axs.ravel()

for i in range(n_data):
    file_data=list_data[i]
    print('data file: ',file_data)
    
#    w1_file='dat_file/g'+file_dat[0:4]+'r_LuS_20180401-now.dat'
#    df_w1=pd.read_csv(w1_file,delim_whitespace=True,header=None,usecols=[0,1,2]) 
    df_data=pd.read_csv(file_data) #,secols=[1,2,3]) 
    print(df_data)

    JD=df_data['JD'].map('{:.5f}'.format).astype(np.float64)
#    print(JD1)
    Mag=df_data[filter_mag[i]]
#    print(R1)
    err=df_data[filter_err[i]]
#    print(eR1)
    

#    axs[i].errorbar(JD0,R0,yerr=eR0,linestyle='--',label='no w',lw=1)
    axs[i].errorbar(JD,Mag,yerr=err,linestyle='--',lw=1)  
    axs[i].set_xlabel('JD')
    axs[i].set_ylabel(filter_mag[i])
    axs[i].set_title(obj_name)
    axs[i].invert_yaxis()
#    axs[i].legend(loc='best')
    


plt.savefig('FU_Orion_plot_BCDE.pdf') 
plt.savefig('FU_Orion_plot_BCDE.png') 

print("... save file as FU_Orion_plot_BCDE.pdf ...")
print("... save file as FU_Orion_plot_BCDE.png ...")
    
