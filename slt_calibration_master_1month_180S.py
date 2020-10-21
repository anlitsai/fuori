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
$ python slt_calibration_science_target.py slt201908
or
$ python slt_calibration_science_target.py slt20190822

"""



#dir_root='/home/altsai/project/20190801.NCU.EDEN/data/gasp/'
#dir_root='/home/altsai/gasp/lulin_data/2019/slt/'
#dir_month='slt201908'
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
yearmonth=sys.argv[1]
#yearmonth='201911'
year=str(yearmonth[0:4])
month=str(yearmonth[4:6])

#folder=sys.argv[1]
#folder='slt201908'
dir_month='slt'+yearmonth
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



'''
def reject_outliers(data,m=2.):
    diff=np.abs(data-np.median(data))
    mdev=np.median(diff)
    s=diff/mdev if mdev else 0
    return data[s<m]

print(reject_outliers(array_each_bias))
'''
'''
def reject_outliers_data(data3d,threshold):
    median_value=np.nanmedian(np.nanmedian(data3d,axis=(1,2)))
    print(median_value)
    diff=abs(data3d-median_value)
#    print(diff)
    med_diff=np.median(diff)
#    print(med_diff)
#    print('...remove outlier...')
    s=diff/med_diff
#    if med_diff == 0:
#        s=0
#    else:
#        s=diff/med_diff
        #print(s)
#    data3d_keep=np.where(s<threshold,data3d,np.NaN)
    data3d_keep=np.where(s<threshold,data3d,median_value)
#    data3d_keep=np.where(s<threshold,data3d,np.NaN)
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data3d_keep
'''
'''
def reject_outliers_data(data3d,threshold):
#    median_value_per_px=np.nanmedian(np.nanmedian(data3d,axis=0))
    median_value_per_px=np.nanmedian(data3d,axis=0)
    print('median_value_per_px',median_value_per_px)
    diff=abs(data3d-median_value_per_px)
#    print(diff)
    med_diff=np.nanmedian(diff)
#    print(med_diff)
#    print('...remove outlier...')
    s=diff/med_diff
    print('diff/med_diff',s)
#    if med_diff == 0:
#        s=0
#    else:
#        s=diff/med_diff
        #print(s)
#    data3d_keep=np.where(s<threshold,data3d,np.NaN)
#    data3d_keep=np.where(s<threshold,data3d,median_value)
    data3d_keep=np.where(s<threshold,data3d,np.NaN)
    bad=np.count_nonzero(np.isnan(data3d_keep))
    px_area=np.count_nonzero(data3d_keep)
    rate=bad/px_area
    print('number of bad/total px: ',bad,'/',px_area,'=',rate)
    return data3d_keep

'''
'''
def reject_outliers_at_same_px(data3d) :#,threshold=3):
    nn=data3d.shape[0]
    xn=data3d.shape[1]
    yn=data3d.shape[2]
    print('number of files: ',nn)
    print('data3d.shape: ',data3d.shape)
#    median_value_per_px=np.nanmedian(np.nanmedian(data3d,axis=0))
    median_value_at_same_px=np.nanmedian(data3d,axis=0)
    print('median_value_at_same_px:',median_value_at_same_px)
    print('median_value_at_same_px.shape:',median_value_at_same_px.shape)
    diff_at_same_px=abs(data3d-median_value_at_same_px)
#    print(diff)
    med_diff=np.nanmedian(diff_at_same_px, axis=0)
    print('med_diff.shape',med_diff.shape)
#    print(med_diff)
#    print('...remove outlier...')
#    s=diff_at_same_px/med_diff if med_diff else 0
    s=np.zeros((nn,xn,yn))
    for i in range (0,xn):
        for j in range (0,yn):
            if med_diff[:][i][j]==0:
                s[:][i][j]=0
            else:
                s[:][i][j]=diff_at_same_px[:][i][j]/med_diff[i][j]
    print('diff/med_diff',s)
#    if med_diff == 0:
#        s=0
#    else:
#        s=diff/med_diff
        #print(s)
#    data3d_keep=np.where(s<threshold,data3d,np.NaN)
#    data3d_keep=np.where(s<threshold,data3d,median_value)
#    for i in range(0,n_files):
    print('s:',s.shape)
    print(s)
    data3d_keep=np.where(s<=3,data3d,np.NaN)   
    bad=np.count_nonzero(np.isnan(data3d_keep))
    px_area=np.count_nonzero(data3d_keep)
    rate=bad/px_area
    print('number of bad/total px: ',bad,'/',px_area,'=',rate)
    return data3d_keep

#array_each_bias_keep=array_each_bias[(np.abs(stats.zscore(array_each_bias)<3).all(axis=0))]
#array_each_bias_keep=array_each_bias[abs()]
'''
'''
def reject_outliers_img_file(data3d,threshold):
    mean_per_img=np.nanmean(data3d,axis=(1,2))
    print(mean_per_img)
    n=mean_per_img.shape[0]
    median_mean_value=np.nanmedian(mean_per_img)
    #print(median_value)
    xx,yy,zz=data3d.shape
    data3d_keep=np.empty((xx,yy,zz))
    data3d_keep[:]=np.nan
    diff=abs(mean_per_img-median_mean_value)
    med_diff=np.median(diff)
    s=diff/med_diff
#    data3d_keep=np.where(s<threshold,data3d,np.nan)
    k=0
    for i in range(n):
        if med_diff == 0:
            s=0
        else:
            s=diff[i]/med_diff
            #print(s)
        if s<threshold:
            data3d_keep[i]=data3d[i]
            k=k+1
    bad=n-k
#    bad=np.count_nonzero(np.isnan(data3d_keep))
    rate=bad/n
    print('number of reject/total image: ',bad,'/',n,'=',rate)
#        data3d_keep[i]=np.where(s<threshold,data3d[i])
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data3d_keep
'''


'''
def reject_outliers_px(data2d,threshold):
    median_value=np.nanmedian(data2d)
    print('median:',median_value)
    diff=abs(data2d-median_value)
#    print('data-median',diff)
    med_diff=np.nanmedian(diff)
#    print('median(data-median)',med_diff)
#    print('...remove outlier...')
    s=diff/med_diff
    print('diff/med_diff',s)
    data2d_keep=np.where(s<threshold,data2d,np.NaN)
#    data2d_keep=np.where(s<threshold,data2d,median_value)
    bad=np.count_nonzero(np.isnan(data2d_keep))
    px_area=np.count_nonzero(data2d_keep)
    rate=bad/px_area
    print('number of bad/total px: ',bad,'/',px_area,'=',rate)
    return data2d_keep
'''


'''
def reject_outliers2_px(data2d,threshold):
#    mode_value=stats.mode(data2d)
#    print('mode:',mode_value)
    mean_value=np.nanmean(data2d)
    print('mean:',mean_value)
    median_value=np.nanmedian(data2d)
    print('median:',median_value)
    diff=abs(data2d-mean_value)
    print('data-median',diff)
    med_diff=np.nanmedian(diff)
#    print('median(data-median)',med_diff)
#    print('...remove outlier...')
    s=diff/med_diff
#    print('diff/med_diff',s)
    data2d_keep=np.where(s<threshold,data2d,np.NaN)
#    data2d_keep=np.where(s<threshold,data2d,mean_value)
    bad=np.count_nonzero(np.isnan(data2d_keep))
    px_area=np.count_nonzero(data2d_keep)
    rate=bad/px_area
    print('number of bad/total px: ',bad,'/',px_area,'=',rate)
#    data3d_interp=data3d_keep.interp
#    print('...interpolation...')
#    x,y=np.indices(data3d_keep[0].shape)
    return data2d_keep
'''

'''
def interp_per_img(data2d):
#    np.count_nonzero(~np.isnan(data))
    # 4194304=2048x2048
#    np.count_nonzero(np.isnan(data))
    # 0   
    x_grid = np.arange(0, data2d.shape[1])
    y_grid = np.arange(0, data2d.shape[0])
    #mask invalid values
    data_bad = np.ma.masked_invalid(data2d)
    xx, yy = np.meshgrid(x_grid, y_grid)
    #get only the valid values
    x1 = xx[~data_bad.mask]
    y1 = yy[~data_bad.mask]
    newdata = data_bad[~data_bad.mask]  
#    print(newdata)
    data_interp = interpolate.griddata((x1, y1), newdata.ravel(), (xx, yy), method='linear')
    return data_interp
'''






print(' ---------------------------')
print(' Master Bias (mean) ')
print(' ---------------------------')


'''

bias_files_ok=[line.rstrip('\n') for line in open('list_bias_ok.txt')]

data_ok=[]
data_ng=[]

for file in bias_files_ok:
    data=fits.open(file)[0].data
    print(data.shape)
    data_cut=reject_outliers_px(data, 10)
    data_ok.append(data_cut)
    print(data.shape)
    filename='cut_'+file
    print(filename)
    hdu=fits.PrimaryHDU(data=data_cut)
    hdu.writeto(filename,overwrite=True)



bias_files_ng=[line.rstrip('\n') for line in open('list_bias_ng.txt')]

for file in bias_files_ng:
    data=fits.open(file)[0].data
    print(data.shape)
    data_cut=reject_outliers_px(data, 10)
    data_ng.append(data_cut)
    print(data.shape)
    filename='cut_'+file
    print(filename)
    hdu=fits.PrimaryHDU(data=data_cut)
    hdu.writeto(filename,overwrite=True)


    
data_ok_mean=np.nanmean(data_ok, axis=0)
data_ng_mean=np.nanmean(data_ng, axis=0)


hdu=fits.PrimaryHDU(data=data_ok_mean)
hdu.writeto('mean_ok.fits',overwrite=True)

hdu=fits.PrimaryHDU(data=data_ng_mean)
hdu.writeto('mean_ng.fits',overwrite=True)



data_mean=np.nanmean(data_all,axis=0)
hdu=fits.PrimaryHDU(data=data_mean)
hdu.writeto('mean.fits',overwrite=True)




'''

#sys.exit(0)

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



#print(array_each_bias.dtype)
#df=pd.DataFrame(array_each_bias)
#pd.to_numeric(array_each_bias, errors='coerce')
#uint16
#pd.to_numeric(df['tester'], errors='coerce')
#pd.to_numeric(array_each_bias, errors='coerce')





#sys.exit(0)

'''
count_nan=np.count_nonzero(np.isnan(array_each_bias))
print('number of bad px: ',count_nan)

count_value=np.count_nonzero(~np.isnan(array_each_bias))
print('number of good px: ',count_value)
print('min,max',np.nanmin(array_each_bias),np.nanmax(array_each_bias))
'''

#print('...start to remove outlier bias...')
#print('...remove outlier image...')
#print('...remove outlier data...')
#bias_keep=reject_outliers_data(array_each_bias,3)
#print(bias_keep)
#print('min,max',np.nanmin(bias_keep),np.nanmax(bias_keep))




print('...generate master bias...')
#master_bias=np.nanmean(array_each_bias, axis=0)
#master_bias=np.nanmean(bias_keep, axis=0)
#median_bias_per_px=np.nanmedian(bias_keep, axis=0)
#print(median_bias_per_px)
#print('min, max',np.nanmin(median_bias_per_px),np.nanmax(median_bias_per_px))

mean_bias=np.mean(array_each_bias, axis=0)
print(mean_bias.shape)
print(mean_bias)

del array_each_bias

#plt.title('Master Bias')
#plt.imshow(master_bias)
#plt.show()

#print('...remove outlier pixel...')
#mean_bias_keep=reject_outliers_px(mean_bias,30)
#mean_bias_keep=mean_bias


'''
print('number of px per img: 2048x2048 =',2048*2048)
count_nan=np.count_nonzero(np.isnan(master_bias))
print('number of bad px: ',count_nan)
count_value=np.count_nonzero(~np.isnan(master_bias))
print('number of good px: ',count_value)
'''

master_bias=mean_bias #_keep
print(master_bias)
print('min,max, mean', np.nanmin(master_bias),np.nanmax(master_bias), np.nanmean(master_bias))
#plt.title('Master Bias')
#plt.imshow(master_bias)
#plt.show()


'''
print('...start to remove outlier bias')
master_bias2=reject_outliers_px(master_bias,30)
#print(master_bias2)
print('number of px per img: 2048x2048 =',2048*2048)
count_nan=np.count_nonzero(np.isnan(master_bias2))
print('number of bad px: ',count_nan)
count_value=np.count_nonzero(~np.isnan(master_bias2))
print('number of good px: ',count_value)
'''

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
#cmd_search_file_dark='find ./ |grep '+dir_month+' | grep fts | grep Dark'
#list_file_dark=os.popen(cmd_search_file_dark,"r").read().splitlines()
#print(list_file_dark)
'''
cmd_search_dark_time='find ./ |grep '+dir_month+' | grep fts | grep Dark | cut -d / -f4 | cut -d - -f3 | cut -d . -f1 | sort | uniq'
list_dark_time=os.popen(cmd_search_dark_time,"r").read().splitlines()
#print(list_dark_time)
'''
cmd_search_dark='find ./ |grep '+dir_month+' | grep fts | grep Dark | grep 180S'
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
fitsname_master_dark='master_dark_180S_'+dir_month+'.fits'
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


#cmd_search_sci_filter="find ./ |grep "+dir_month+" |grep fts | grep flat | cut -d / -f5 | awk -F'_Astro' '{print $1}'| rev | cut -c1-3 | rev | cut -d - -f2 | sort | uniq"
#cmd_search_sci_filter="find ./ |grep "+dir_month+" |grep GASP| grep fts | rev | cut -d - -f1 | cut -d _ -f3 | rev| sort | uniq"
cmd_search_sci_filter="find ./ |grep "+dir_month+" | grep GASP | cut -d / -f6 | grep fts|cut -d '@' -f2 | cut -d _ -f1 | cut -d - -f2 | sort | uniq"
#cmd_search_sci_filter="find ./ |grep "+dir_month+" | grep GASP | cut -d / -f7 | grep fts|cut -d '@' -f2 | cut -d _ -f1 | cut -d - -f2 | sort | uniq"

print(cmd_search_sci_filter)
list_flat_filter=os.popen(cmd_search_sci_filter,"r").read().splitlines()
print('all filter: ',list_flat_filter)
#print('all filter: ',list_flat_filter)


#list_flat_filter=['R']
for i in list_flat_filter:
    print('filter',i)


#sys.exit(0)



master_flat={}
#print(master_flat)
#awk -F'PANoRot-' '{print $2}'|cut -d _ -f1
for i in list_flat_filter:
    cmd_search_file_flat='find ./ |grep '+dir_month+' | grep fts | grep flat | grep PANoRot-'+i
    print(cmd_search_file_flat)
    list_file_flat=os.popen(cmd_search_file_flat,"r").read().splitlines()
    print('filter: ', i)
    print('file list',list_file_flat)
#    print(len(list_file_flat))
    #array_flat=np.array([pyfits.getdata(j) for j in list_file_flat])
    array_flat=np.array([fits.open(j)[0].data for j in list_file_flat])
#    print(array_flat.shape)
#    print('...remove outlier data...')
#    flat_keep=reject_outliers_at_same_px(array_flat)
#    flat_keep2=reject_outliers_data(flat_keep,par2)
    print('...generate master flat '+i+'...')
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
    master_flat[i]=norm_mean_flat
#        print(master_flat[idx_filter_time])
#    print(mean_flat_each_filter[1000][1000])
#    plt.title('Master Flat '+i)
#    plt.imshow(mean_flat_each_filter)
#    plt.show()
    print('...output master flat '+i+' to fits file...')
    fitsname_master_flat='master_flat_'+i+'_180S_'+dir_month+'.fits'
    hdu=fits.PrimaryHDU(master_flat[i])
#        now=str(datetime.now())  
#        fits.header.add_history('Master Flat generated at '+now+' UTC')
    hdu.writeto(dir_master+fitsname_master_flat,overwrite=True)
#        imhead.add_history('Master bias, dark are applied at '+now+' UTC')
#        fits.writeto(fitsname_master_flat,data=norm_mean_flat_each_filter,header=imhead,overwrite=True)

del list_flat_filter
del list_file_flat
del array_flat

print('... finished ...')

'''
print('...move master files to '+dir_master+'...')
#cmd_search_master_files="find ./ -type f -name 'master*'"
cmd_search_master_files="find ./ |grep '^\./master_'"
list_master_files=os.popen(cmd_search_master_files,"r").read().splitlines()
print(list_master_files)


for i in list_master_files:
    cmd_mv_file="mv "+i+" ./"+dir_master
    list_master_files=os.popen(cmd_mv_file,"r")

print('...move ./'+dir_master+' to ./'+yearmonth+'/ ...')
if os.path.exists(yearmonth+'/'+dir_master):
    shutil.rmtree(yearmonth+'/'+dir_master)
cmd_mv_dir='mv ./'+dir_master+' ./'+yearmonth+'/'
mv_dir=os.popen(cmd_mv_dir,"r")

    
del list_master_files
'''
