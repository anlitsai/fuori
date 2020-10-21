# Step-by-step of how to run scripts for data taken every month

#a='202003'  
a=$1

## step0
search_no_wcs.py $a  
#upload no_wcs_files to ycc and modify them  

## step1	
for i in $a;do   
  echo "python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log"  
  python slt_calibration_master_1month_180S.py $i | tee slt_calibration_master_1month_180S_$i.log  
done  


## step2
b=`find ./|grep $a[0-3][0-9]$|cut -d / -f3|sort|uniq`   
echo $b   
for j in $b;do  
  echo $j  
  echo "python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log"  
  python slt_calibration_science_calibration_180S.py $j | tee slt_calibration_science_calibration_180S_$j.log  
done  


## step3
python gasp_target_fitsheader_info_exclude_baddata_permonth.py $a  


## step4
#manually modify this file:  
python gasp_target_fitsheader_info_exclude_baddata_join.py  


## step5
c=`cat check_science_target_list.txt`  

year=`echo $a|cut -c-4`  
month=`echo $a|cut -c5-6`  
ym=`echo $year'-'$month`  

last_date=`cat gasp_target_fitsheader_info_exclude_baddata_join.txt | grep $ym|tail -1| cut -d "|" -f2| cut -d - -f3`  

d1=$a'01'  
d2=$a$last_date  

for i in $c;do python Rmag_aperture_annulus_r_file_median_w1_subplot_date_target.py $d1 $d2 $i | tee 'Rmag_aperture_annulus_r_file_median_w1_subplot_'$d1'-'$d2'_'$i'.log';done  

## final_step
#modify the dir  
./gasp_target_result_join.sh  
python gasp_target_result_join_Rmag_JD.py  


