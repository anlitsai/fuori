
echo "=== slt ==="
echo "slt Bias "
cat check_exposure_pixel_slt_Bias.log|cut -f3|sort|uniq
# 0S

echo "slt Dark "
cat check_exposure_pixel_slt_Dark.log|cut -f3|sort|uniq
# 2S, 5S, 10S, 15S, 20S, 30S

echo "---------"
echo "slt FUOri B "
cat check_exposure_pixel_slt_FUOri.log|grep B_|cut -f3|sort|uniq
# 10S, 20S, 30S

#echo "slt flat B "
#cat check_exposure_pixel_slt_Flat.log|grep B_|cut -f3|sort|uniq
# 1S

echo "slt FUOri V "
cat check_exposure_pixel_slt_FUOri.log|grep V_|cut -f3|sort|uniq
# 5S, 10S, 15S

#echo "slt flat V "
#cat check_exposure_pixel_slt_Flat.log|grep V_|cut -f3|sort|uniq
# 1S

echo "slt FUOri R "
cat check_exposure_pixel_slt_FUOri.log|grep R_|cut -f3|sort|uniq
# 5S, 10S, 15S

#echo "slt flat R "
#cat check_exposure_pixel_slt_Flat.log|grep R_|cut -f3|sort|uniq
# 1S, 3S, 4S

echo "=== LOT ==="
echo "LOT Bias "
cat check_exposure_pixel_LOT_Bias.log|cut -f3|sort|uniq
# 0S

echo "LOT Dark "
cat check_exposure_pixel_LOT_Dark.log|cut -f3|sort|uniq
# 1S, 2S, 3S, 5S, 10S, 30S

echo "---------"
echo "LOT FUOri B "
cat check_exposure_pixel_LOT_FUOri.log|grep B_|cut -f3|sort|uniq
# 10S

#echo "LOT flat B "
#cat check_exposure_pixel_LOT_flat.log|grep B_|cut -f3|sort|uniq
# 5S, 10S 


echo "LOT FUOri V "
cat check_exposure_pixel_LOT_FUOri.log|grep V_|cut -f3|sort|uniq
# 3S, 5S

#echo "LOT flat V "
#cat check_exposure_pixel_LOT_flat.log|grep V_|cut -f3|sort|uniq
# 5S 


echo "LOT FUOri R "
cat check_exposure_pixel_LOT_FUOri.log|grep R_|cut -f3|sort|uniq
# 1S, 5S

#echo "LOT flat R "
#cat check_exposure_pixel_LOT_flat.log|grep R_|cut -f3|sort|uniq
# 5S 

echo "==========="





