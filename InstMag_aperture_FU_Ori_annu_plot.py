#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 03:49:10 2020

@author: altsai
"""

import os
import sys
import shutil
import numpy as np
import csv
import time
import math
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates
#from astropy.coordinates import ICRS, Galactic, FK4, FK5 # Low-level frames
#from astropy.coordinates import Angle, Latitude, Longitude  # Angles
#from astropy.coordinates import match_coordinates_sky
from astropy.table import Table
from photutils import CircularAperture
from photutils import SkyCircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
from photutils import SkyCircularAnnulus
# https://photutils.readthedocs.io/en/stable/aperture.html
#from phot import aperphot
# http://www.mit.edu/~iancross/python/phot.html

import matplotlib.pyplot as plt
import matplotlib.axes as ax
from astropy.io import fits
from astropy.wcs import WCS
#from photutils import DAOStarFinder
#from astropy.stats import mad_std
# https://photutils.readthedocs.io/en/stable/getting_started.html

from numpy.polynomial.polynomial import polyfit
#from astropy.stats import sigma_clipped_stats
#from photutils.psf import IterativelySubtractedPSFPhotometry
#from statistics import mode
#from astropy.visualization import simple_norm
#from photutils.utils import calc_total_error
#from astropy.stats import mad_std
#import matplotlib.gridspec as gridspec

#import julian
#from datetime import datetime
#from datetime import timedelta
#from datetime import date
#import datetime

file_instmagB='./InstMag_Bmag/annu_w1_20201005-20201015/FU_Ori/Bmag_aperture_FU_Ori_annu.txt'
file_instmagV='./InstMag_Vmag/annu_w1_20201005-20201015/FU_Ori/Vmag_aperture_FU_Ori_annu.txt'
file_instmagR='./InstMag_Rmag/annu_w1_20201005-20201015/FU_Ori/Rmag_aperture_FU_Ori_annu.txt'

df_instmagB=pd.read_csv(file_instmagB,sep='|')
JD_B=df_instmagB['JD'].tolist()
#idx_refstar=df_refstar[df_refstar['ObjectName'].str.contains(obj_name)].index.tolist()
#idx_refstar=df_refstar[df_refstar['ObjectName'].str.match(obj_name)].index.tolist()
#print('JD : ', JD)
df_instmagV=pd.read_csv(file_instmagV,sep='|')
JD_V=df_instmagV['JD'].tolist()
df_instmagR=pd.read_csv(file_instmagR,sep='|')
JD_R=df_instmagR['JD'].tolist()

InstMagB_FU_Ori=df_instmagB['InstMag_FU_Ori'].tolist()
InstMagB_A=df_instmagB['InstMag_A'].tolist()
InstMagB_B=df_instmagB['InstMag_B'].tolist()
InstMagB_C=df_instmagB['InstMag_C'].tolist()
InstMagB_D=df_instmagB['InstMag_D'].tolist()
InstMagB_E=df_instmagB['InstMag_E'].tolist()

InstMagV_FU_Ori=df_instmagV['InstMag_FU_Ori'].tolist()
InstMagV_A=df_instmagV['InstMag_A'].tolist()
InstMagV_B=df_instmagV['InstMag_B'].tolist()
InstMagV_C=df_instmagV['InstMag_C'].tolist()
InstMagV_D=df_instmagV['InstMag_D'].tolist()
InstMagV_E=df_instmagV['InstMag_E'].tolist()

InstMagR_FU_Ori=df_instmagR['InstMag_FU_Ori'].tolist()
InstMagR_A=df_instmagR['InstMag_A'].tolist()
InstMagR_B=df_instmagR['InstMag_B'].tolist()
InstMagR_C=df_instmagR['InstMag_C'].tolist()
InstMagR_D=df_instmagR['InstMag_D'].tolist()
InstMagR_E=df_instmagR['InstMag_E'].tolist()


fig=plt.figure(figsize=(6,8))
#fig,axs=plt.subplots(2,1,sharex=True) #,gridspec_kw={'hspace':0})


plt.subplot(311)
plt.subplots_adjust(right=0.75)
plt.gca().invert_yaxis()

fig.suptitle('InstMag(B) vs. JD')
#plt.xlabel('JD')
plt.ylabel('InstMag(B)')
plt.scatter(JD_B,InstMagB_FU_Ori) #, label='FU_Ori') 
plt.scatter(JD_B,InstMagB_A, label='A') 
plt.scatter(JD_B,InstMagB_B, label='B') 
plt.scatter(JD_B,InstMagB_C, label='C') 
plt.scatter(JD_B,InstMagB_D, label='D') 
plt.scatter(JD_B,InstMagB_E, label='E') 

plt.legend(loc='right')


plt.subplot(312)
plt.subplots_adjust(right=0.75)
plt.gca().invert_yaxis()

fig.suptitle('InstMag(V) vs. JD')
#plt.xlabel('JD')
plt.ylabel('InstMag(V)')
plt.scatter(JD_V,InstMagV_FU_Ori) #, label='FU_Ori') 
plt.scatter(JD_V,InstMagV_A, label='A') 
plt.scatter(JD_V,InstMagV_B, label='B')  
plt.scatter(JD_V,InstMagV_C, label='C')  
plt.scatter(JD_V,InstMagV_D, label='D') 
plt.scatter(JD_V,InstMagV_E, label='E') 

plt.legend(loc='right')




plt.subplot(313)
plt.subplots_adjust(right=0.75)
plt.gca().invert_yaxis()

fig.suptitle('InstMag(R) vs. JD')
plt.xlabel('JD')
plt.ylabel('InstMag(R)')
plt.scatter(JD_R,InstMagR_FU_Ori) #, label='FU_Ori') 
plt.scatter(JD_R,InstMagR_A, label='A')  
plt.scatter(JD_R,InstMagR_B, label='B') 
plt.scatter(JD_R,InstMagR_C, label='C')  
plt.scatter(JD_R,InstMagR_D, label='D') 
plt.scatter(JD_R,InstMagR_E, label='E') 

plt.legend(loc='right')


#plt.savefig(dir_obj+'Rmag_JD_'+obj_name+'_RefStar_annu.png',dpi=200)
#plt.savefig(dir_obj+'Rmag_JD_'+obj_name+'_RefStar_annu.pdf')

#plt.close()





