#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:23:32 2019

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
from astropy.stats import sigma_clipped_stats
from photutils.psf import IterativelySubtractedPSFPhotometry
from statistics import mode
from astropy.visualization import simple_norm
from photutils.utils import calc_total_error

month=sys.argv[1]
cmd_search_file_sci='find ./'+month+' |grep FU_Ori|grep fts|sort'
#cmd_search_file_sci='find ./20191? |grep FU_Ori|grep fts|sort'
#cmd_search_file_sci='find ./201??? |grep FU_Ori|grep fts|sort'
#print(cmd_search_file_sci)
list_file_sci=os.popen(cmd_search_file_sci,"r").read().splitlines()
#print(list_file_sci)
n_file_sci=len(list_file_sci)

outfile='search_no_wcs_'+month+'_FU_Ori.txt'
f=open(outfile,'w')

msg0='... total '+str(n_file_sci)+' files ...'
print(msg0)
f.write(msg0+'\n')


k=0
for i in range(n_file_sci):
    path_file=list_file_sci[i]
#    print(path_file)
    path=path_file.rsplit('/',1)[0]
    filename=path_file.rsplit('/',1)[1]
#    print(str(i)+') filename: '+filename)
    hdu=fits.open(path_file)[0]
    imhead=hdu.header
#    ra_hhmmss=imhead['RA']
#    dec_ddmmss=imhead['Dec']
#    ra_deg=imhead['CRVAL1']
#    dec_deg=imhead['CRVAL2']
    try: 
        ra_deg=imhead['CRVAL1']
    except KeyError:
        k=k+1
        msg1=str(k)+') '+path_file
#        msg1=path_file
        print(msg1)
        f.write(msg1+'\n')

msg2='... there are '+str(k)+' files have no wcs coordinate ...'
print(msg2)
f.write(msg2+'\n')
msg3='... write file to '+outfile+' ...'
print(msg3)
f.write(msg3+'\n')

f.close()
