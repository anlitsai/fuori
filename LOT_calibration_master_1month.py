#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 04:48:28 2019

@author: altsai
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

generate master bias, master dark, master flat for one month.
$ condaa
$ python LOT_calibration_science_target.py LOT201908
or
$ python LOT_calibration_science_target.py LOT20190822

"""



#dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
#dir_root='/home/altsai/gasp/lulin_data/2019/LOT/'
#dir_month='LOT201908'
#date=dir_month+'22'
#dir_master=dir_month+'_master/'
#dir_calib_sci=date+'_calib_sci/'


import os
import sys
import shutil
#import re
import numpy as np
#import numpy
from astropy.io import fits
#import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
import pandas as pd
from datetime import datetime
#from scipy import interpolate
#from scipy import stats



#print("Which Month you are going to process ?")
#yearmonth=input("Enter a year-month (ex: 201908): ")
#yearmonth=sys.argv[1]
yearmonth='202010'
year=str(yearmonth[0:4])
month=str(yearmonth[4:6])

#folder=sys.argv[1]
#folder='LOT201908'
dir_month='LOT'+yearmonth
#print(dir_month)
dir_master=yearmonth+'/'+dir_month+'_master/'
#dir_master='data/'+yearmonth+'/'+dir_month+'_master/'

print(dir_master)
#dir_calib_sci=date+'_calib_sci/'
#print(dir_calib_sci)

if os.path.exists(dir_master):
    shutil.rmtree(dir_master)
os.makedirs(dir_master,exist_ok=True)

print('...generate master files on '+dir_month+'...')

#sys.exit(0)


'''
logfile=dir_month+'_master.log'
sys.stdout=open(logfile,'w')
print(sys.argv)
'''

#time_calib_start=strftime("%Y-%m-%d %H:%M:%S", gmtime())
#time_calib_start=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_calib_start=str(datetime.now())  
print('Data calibrated by An-Li Tsai at '+time_calib_start+' UTC')

print(' ---------------------------')
print(' Master Bias (mean) ')
print(' ---------------------------')


array_each_bias=[]

#cmd_search_file_bias='find ./ |grep '+dir_month+' | grep fts | grep Bias'
cmd_search_file_bias='find ./'+yearmonth+'/|grep '+dir_month+' | grep fts | grep Bias'
list_file_bias=os.popen(cmd_search_file_bias,"r").read().splitlines()
print(list_file_bias)
n_bias=len(list_file_bias)
print('number of total bias:',n_bias)
#sys.exit(0)
#array_each_bias=np.array([pyfits.getdata(i) for i in list_file_bias])
#array_each_bias=np.array([fits.open(i)[0].data for i in list_file_bias])
n_bias_2048=0
for i in range(n_bias):
    j=list_file_bias[i]
#    print(j)
    imdata=fits.open(j)[0].data
    imhead=fits.open(j)[0].header
    nx=imhead['NAXIS1']
#    print('NAXIS1',nx)
    if nx==2048:
        array_each_bias.append(imdata)
        n_bias_2048=n_bias_2048+1
array_each_bias=np.array(array_each_bias,dtype=int)
print(array_each_bias)
#print(type(array_each_bias))

del list_file_bias

print('number of selected bias:',n_bias_2048)

#print(array_each_bias.shape)
#n_arr_bias=(array_each_bias.shape[0])
#n_arr_bias=len(array_each_bias)
print('number of total px: 2048x2048x',n_bias_2048,' = ', 2048*2048*n_bias_2048)

#sys.exit(0)

print('...generate master bias...')

mean_bias=np.mean(array_each_bias, axis=0)
print(mean_bias.shape)
print(mean_bias)

del array_each_bias


master_bias=mean_bias #_keep
print(master_bias)
print('min,max, mean', np.nanmin(master_bias),np.nanmax(master_bias), np.nanmean(master_bias))
#plt.title('Master Bias')
#plt.imshow(master_bias)
#plt.show()

print('...output master bias to fits file...')

fitsname_master_bias='master_bias_'+dir_month+'.fits'
hdu=fits.PrimaryHDU(data=master_bias)
#hdr=fits.Header()
#now=str(datetime.now())  
#hdr.add_history('Master Bias generated at '+now+' UTC')
#hdu._writeheader('Master Bias generated at '+now+' UTC')
hdu.writeto(dir_master+fitsname_master_bias,overwrite=True)
#now=str(datetime.now())  
#imhead.add_history('Master bias is generated at '+now+' UTC')
#fits.writeto(fitsname_master_bias,data=master_bias,header=imhead,overwrite=True)

#sys.exit(0)

print(' ---------------------------')
print(' Master Dark (subtract from Bias) ')
print(' ---------------------------')


bias_list=['001S','003S','005S','010S']

for bias_time in bias_list:


    cmd_search_dark='find ./ |grep '+dir_month+' | grep fts | grep Dark | grep '+bias_time
    print(cmd_search_dark)
    list_file_dark=os.popen(cmd_search_dark,"r").read().splitlines()
    print(list_file_dark)

    #sys.exit(0)

    #print('...start to remove outlier dark...')

    #master_dark={}

    array_dark=np.array([fits.open(j)[0].data for j in list_file_dark])
    #print('...remove outlier data...')
    #dark_keep=reject_outliers_data(array_dark,par1)
    #    dark_each_time_keep2=reject_outliers_data(dark_each_time_keep,3)
    #    print(dark_keep)
    print('...generate master dark...')
    dark_subtract=array_dark-master_bias
    mean_dark=np.mean(dark_subtract,axis=0)
    #print('...remove outlier pixel...')
    #mean_dark_keep=reject_outliers_px(mean_dark,par2)
    master_dark=mean_dark
    #print('skip this step')
    #    master_dark[i]=mean_dark_each_time_keep
    #    print(master_dark_each_time[1000][1000])
    #    plt.title('Master Dark '+i)
    #    plt.imshow(master_dark_each_time)
    #    plt.show()
    print('...output master dark to fits file...')
    fitsname_master_dark='master_dark_'+bias_time+'_'+dir_month+'.fits'
    now=str(datetime.now())  
    #    fits.header.add_history('Master Dark generated at '+now+' UTC')
    #    hdu=fits.PrimaryHDU(master_dark[i])
    hdu=fits.PrimaryHDU(master_dark)
    hdu.writeto(dir_master+fitsname_master_dark,overwrite=True)
    #    now=str(datetime.now())  
    #    imhead.add_history('Master bias is applied at '+now+' UTC')
    #    fits.writeto(fitsname_master_dark,data=master_dark_each_time,header=imhead,overwrite=True)

    del list_file_dark
    del array_dark





#sys.exit(0)

print(' ---------------------------')
print(' Master Flat (subtract from Dark and Bias) ')
print(' ---------------------------')

#os.chdir(dir_date+"/flat/")
#cmd_search_sci_filter="find ./ |grep "+dir_month+" | grep GASP | cut -d / -f6 | grep fts|cut -d '@' -f2 | cut -d _ -f1 | cut -d - -f2 | sort | uniq"

#print(cmd_search_sci_filter)
#list_flat_filter=os.popen(cmd_search_sci_filter,"r").read().splitlines()
#print('all filter: ',list_flat_filter)

#list_flat_filter=['R']
#for i in list_flat_filter:
#    print('filter',i)


#sys.exit(0)

list_flat_filter=['V','B','R']


master_flat={}
#print(master_flat)
#awk -F'PANoRot-' '{print $2}'|cut -d _ -f1
for flat_filter in list_flat_filter:
#    cmd_search_file_flat='find ./ |grep '+dir_month+' | grep fits | grep flat | grep PANoRot-'+flat_filter
    cmd_search_file_flat='find ./ |grep '+dir_month+' | grep fits | grep flat | grep '+flat_filter

    print(cmd_search_file_flat)
    list_file_flat=os.popen(cmd_search_file_flat,"r").read().splitlines()
    print('filter: ', flat_filter)
    print('file list',list_file_flat)
#    print(len(list_file_flat))
    #array_flat=np.array([pyfits.getdata(j) for j in list_file_flat])
    array_flat=np.array([fits.open(j)[0].data for j in list_file_flat])
#    print(array_flat.shape)
#    print('...remove outlier data...')
#    flat_keep=reject_outliers_at_same_px(array_flat)
#    flat_keep2=reject_outliers_data(flat_keep,par2)
    print('...generate master flat '+flat_filter+'...')
    print('master bias: ', master_bias.shape)
    print('master dark: ', master_dark.shape)
    print('array flat: ',array_flat.shape) 
#    mean_flat=np.nanmean(flat_keep-master_bias-master_dark,axis=0)  
    mean_flat=np.mean(array_flat,axis=0)  
#        print(np.amax(mean_flat_each_filter))
#    print('...remove outlier pixel...')
#    mean_flat_keep=reject_outliers2_px(mean_flat,par3)
    min_value_flat=np.nanmin(mean_flat)
    max_value_flat=np.nanmax(mean_flat)
    mean_value_flat=np.mean(mean_flat)
    print('min, max =',min_value_flat,max_value_flat)
    flat_subtract=mean_flat-master_bias-master_dark
    #norm_mean_flat=(mean_flat-min_value)/(max_value-min_value)
#    flat_subtract=mean_flat-master_bias-master_dark
#    norm_mean_flat=(mean_flat-min_value)/(max_value-min_value)  #max_value
    norm_mean_flat=mean_flat/mean_value_flat  #normalized to mean value
#        print(np.amax(norm_mean_flat_each_filter))
    master_flat[flat_filter]=norm_mean_flat
#        print(master_flat[idx_filter_time])
#    print(mean_flat_each_filter[1000][1000])
#    plt.title('Master Flat '+i)
#    plt.imshow(mean_flat_each_filter)
#    plt.show()
    print('...output master flat '+flat_filter+' to fits file...')
#    fitsname_master_flat='master_flat_'+flat_filter+'_180S_'+dir_month+'.fits'
    fitsname_master_flat='master_flat_'+flat_filter+'_'+dir_month+'.fits'
    hdu=fits.PrimaryHDU(master_flat[flat_filter])
#        now=str(datetime.now())  
#        fits.header.add_history('Master Flat generated at '+now+' UTC')
    hdu.writeto(dir_master+fitsname_master_flat,overwrite=True)
#        imhead.add_history('Master bias, dark are applied at '+now+' UTC')
#        fits.writeto(fitsname_master_flat,data=norm_mean_flat_each_filter,header=imhead,overwrite=True)

del list_flat_filter
del list_file_flat
del array_flat

print('... finished ...')


