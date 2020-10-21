#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:54:00 2019

@author: altsai
"""

"""
Spyder Editor

check exposure time for each target (with full path).
$ condaa
$ python check_exposure_pixel_file.py _PATH_FILE_

"""

import os
import sys
import shutil
#import re
import numpy as np
#import numpy
from astropy.io import fits
#import pyfits
import matplotlib.pyplot as plt
#import scipy.ndimage as snd
#import glob
#import subprocess
#from scipy import interpolate
#from scipy import stats
#from scipy.interpolate import griddata
#from time import gmtime, strftime
#import pandas as pd
from datetime import datetime

#print()
#print('format: python check_exposure_pixel.py slt20201010')
#print()

path_file=sys.argv[1]


hdu=fits.open(path_file)[0]
imhead=hdu.header
imdata=hdu.data
exptime=imhead['EXPTIME']
idx_time=str(int(exptime))+'S'
imtype=imhead['IMAGETYP']
obj=imhead['OBJECT']
nx1=imhead['NAXIS1']
nx2=imhead['NAXIS2']
try:
    filt=imhead['FILTER']
except:
    filt="N/A"


print(path_file, "\t", str(nx1)+"x"+str(nx2),"\t",idx_time,"\t",str(imtype),"\t",str(obj),"\t", str(filt))
 
#print(' ---------------------------')

#print()


