#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 03:13:49 2019

@author: altsai
"""

import os
import sys
import numpy as np
import csv
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.table import Table
from astropy.wcs import WCS

file_list_in='slt_RefStar_ABE.list'
#print(file_list_in)
#df_list=pd.read_csv(file_list_in,sep='\t',skiprows=[0,1,2,17,18])
df_list=pd.read_csv(file_list_in,sep='\t',skiprows=[0,1,2,19,20])
print(df_list)
#sys.exit(0)

df_list_recommend=df_list.loc[(df_list['NoteIdx']<3) & (df_list['NoteIdx']>-1)].reset_index(drop=True)
n_recommend=len(df_list_recommend)
#sys.exit(0)

regname=['']*n_recommend
obj_ra_hhmmss=['']*n_recommend
obj_dec_ddmmss=['']*n_recommend
obj_r_as=['']*n_recommend
obj_r_inner_as=['']*n_recommend
obj_r_outer_as=['']*n_recommend
star_ra_hhmmss=['']*n_recommend
star_dec_ddmmss=['']*n_recommend
star_ra_deg=[0]*n_recommend
star_dec_deg=[0]*n_recommend
#star_ra_hhmmssdius=['']*n_recommend
#sys.exit(0)

for i in range(n_recommend):
    obj_name=df_list_recommend['ObjectName'][i]
#    regfile='RefStar_'+obj_name+'_fk5.reg'
    regfile='LOT_RefStar_'+obj_name+'_annu_mod_fk5.reg'
    regname[i]=regfile
#    print(regname[i])  
    obj_ra_hhmmss[i]=df_list_recommend['RA_hhmmss'][i].replace(' ',':')
    obj_dec_ddmmss[i]=df_list_recommend['DEC_ddmmss'][i].replace(' ',':')
#    print(obj_ra_hhmmss[i],obj_dec_ddmmss[i])
    id_star=df_list_recommend['RefStarID'][i]    
    print(id_star)
    cmd_search_refstar_radecr='cat '+regfile+' |grep "text={'+id_star+'}"| grep circle | cut -d "(" -f2 | cut -d ")" -f1'
    print(cmd_search_refstar_radecr)
    radecr=os.popen(cmd_search_refstar_radecr,"r").read().splitlines()[0].split(',')
    print(radecr)

#    obj_ra_hhmmss[i]=radecr[0]
#    obj_dec_ddmmss[i]=radecr[1]
    obj_r_as[i]=radecr[2].split('"')[0]
    print(obj_ra_hhmmss[i],obj_dec_ddmmss[i],obj_r_as[i])

    cmd_search_refstar_rinnout='cat '+regfile+' |grep "text={'+id_star+'}"| grep annulus | cut -d "(" -f2 | cut -d ")" -f1'
    print(cmd_search_refstar_rinnout)
    rinnout=os.popen(cmd_search_refstar_rinnout,"r").read().splitlines()[0].split(',')
    print(rinnout)

#    ra=radecr[0]
#    dec=radecr[1]
    obj_r_inner_as[i]=rinnout[2].split('"')[0]
    obj_r_outer_as[i]=rinnout[3].split('"')[0]
    print(obj_r_inner_as[i],obj_r_outer_as[i])


    ra_hhmmss=radecr[0]
    dec_ddmmss=radecr[1]
#    radius=radecr[2].split('"')[0]
#    print(ra,dec,radius)
    star_ra_hhmmss[i]=ra_hhmmss
    star_dec_ddmmss[i]=dec_ddmmss
#    star_ra_hhmmssdius[i]=radius
#    print(ra_hhmmss,dec_ddmmss)
    star_skycoord=SkyCoord(ra_hhmmss,dec_ddmmss,frame='icrs',unit=(u.hourangle,u.deg))
#    print(star_skycoord)
    star_ra_deg[i]=star_skycoord.ra.deg
    star_dec_deg[i]=star_skycoord.dec.deg
#    print(star_skycoord.ra.deg)
#    print(star_skycoord.dec.deg)

#sys.exit(0)
df_list_recommend['RA_hhmmss']=obj_ra_hhmmss
df_list_recommend['DEC_ddmmss']=obj_dec_ddmmss
df_list_recommend['Redshift']=df_list_recommend['Redshift'].map('{:,.4f}'.format)
df_list_recommend.insert(5,'RefStarregname',regname)
df_list_recommend.insert(7,'RefStarRA_hhmmss',star_ra_hhmmss)
df_list_recommend.insert(8,'RefStarDEC_ddmmss',star_dec_ddmmss)
#df_list_recommend.insert(9,'RefStarRadius_as',star_ra_hhmmssdius)
df_list_recommend.insert(9,'RefStarRA_deg',star_ra_deg)
df_list_recommend.insert(10,'RefStarDEC_deg',star_dec_deg)
df_list_recommend.insert(11,'R_circle_as',obj_r_as)
df_list_recommend.insert(12,'R_inner_as',obj_r_inner_as)
df_list_recommend.insert(13,'R_outer_as',obj_r_outer_as)
df_list_recommend['RefStarRA_deg']=df_list_recommend['RefStarRA_deg'].map('{:.4f}'.format)
df_list_recommend['RefStarDEC_deg']=df_list_recommend['RefStarDEC_deg'].map('{:.4f}'.format)
df_list_recommend['Rmag']=df_list_recommend['Rmag'].map('{:.3f}'.format)
df_list_recommend['Rmag_err']=df_list_recommend['Rmag_err'].map('{:.3f}'.format)
df_list_recommend['Vmag']=df_list_recommend['Vmag'].map('{:.3f}'.format)
df_list_recommend['Vmag_err']=df_list_recommend['Vmag_err'].map('{:.3f}'.format)


print(df_list_recommend)

file_list_out='LOT_refStar_radec_annu_ABE.txt'
df_list_recommend.to_csv(file_list_out,sep='|')

print('... write file to: ./'+file_list_out)
print('... finished ...')
