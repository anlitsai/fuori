a=`find ./ |grep slt|grep Dark|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_slt_Dark.log

a=`find ./ |grep slt|grep Bias|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_slt_Bias.log

a=`find ./ |grep slt|grep Flat|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_slt_Flat.log

a=`find ./ |grep slt|grep FU|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_slt_FUOri.log


a=`find ./ |grep LOT|grep Dark|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_LOT_Dark.log

a=`find ./ |grep LOT|grep Bias|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_LOT_Bias.log

a=`find ./ |grep LOT|grep flat|grep fits`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_LOT_flat.log

a=`find ./ |grep LOT|grep FU|grep fts`
for i in $a; do python check_exposure_pixel_file.py $i;done | tee check_exposure_pixel_LOT_FUOri.log


