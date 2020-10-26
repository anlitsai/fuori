
file1='InstMag_apeture_FU_Ori_annu.txt'
file2='BVR_apeture_FU_Ori_annu.txt'
file3='BVR_apeture_FU_Ori_annu_sort.txt'

cat ./InstMag_Bmag/annu_w1_20201005-20201015/FU_Ori/Bmag_aperture_FU_Ori_annu.txt |head -1 |cut -d '|' -f3-15,13,22-25 > $file1

cat ./InstMag_Bmag/annu_w1_20201005-20201015/FU_Ori/Bmag_aperture_FU_Ori_annu.txt |grep fits |cut -d '|' -f3-15,13,22-25 > $file2
cat ./InstMag_Vmag/annu_w1_20201005-20201015/FU_Ori/Vmag_aperture_FU_Ori_annu.txt |grep fits |cut -d '|' -f3-15,13,22-25 >> $file2
cat ./InstMag_Rmag/annu_w1_20201005-20201015/FU_Ori/Rmag_aperture_FU_Ori_annu.txt |grep fits |cut -d '|' -f3-15,13,22-25 >> $file2

cat $file2 | sort -t'|' -k12  > $file3

cat $file3  >> $file1

rm -rf $file2 $file3


file4='InstMag_apeture_FU_Ori_annu_B.txt'
file5='InstMag_apeture_FU_Ori_annu_V.txt'
file6='InstMag_apeture_FU_Ori_annu_R.txt'

cat ./InstMag_Bmag/annu_w1_20201005-20201015/FU_Ori/Bmag_aperture_FU_Ori_annu.txt |grep fits|cut -d '|' -f2-5,13-15,20-21
