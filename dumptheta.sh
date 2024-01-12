#!/bin/bash

X0=-10
nX=10 #10
dX=-10
XMAX=$((X0+nX*dX))
echo " XMAX=" $XMAX
Theta0XZ=10
nThetaXZ=5 #10 or 15
dThetaXZ=1
ThetaXZMAX=$((Theta0XZ+nThetaXZ*dThetaXZ))
echo " ThetaXZMAX=" $ThetaXZMAX

fcldir=/sbnd/data/users/ebatista/imaging-handscan/XYMuon/fcl
rootdir=/sbnd/data/users/ebatista/imaging-handscan/XYMuon/root
gzdir=/sbnd/data/users/ebatista/imaging-handscan/XYMuon/gz
celltreedir=/sbnd/data/users/ebatista/imaging-handscan/XYMuon/celltree
g4fcl=$fcldir/standard_g4_sbnd.fcl
celltreefcl=$fcldir/celltree_sbnd.fcl
wcsimspfcl=$fcldir/wirecell_sim_sp_sbnd-mod.fcl
wcimgfcl=$fcldir/wcls-sig-to-img_v4.fcl
imgbookkeeping=$gzdir/README
event=0

# input stage to run
stage=none
while [[ "$stage" != "gen" && "$stage" != "g4" && "$stage" != "wcsimsp" && "$stage" != "wcimg" ]]
do
  printf "\n Stage to run? [gen] [g4] [wcsimsp] [wcimg]: "
  read stage
done


for X in `seq $X0 $dX $XMAX`; do
  #echo "X=$X:";
  for Theta in `seq $Theta0XZ $dThetaXZ $ThetaXZMAX`; do
    #echo "Theta=$Theta";



    angle=$Theta
    printf "\n angle=$angle\n"
    #touch $gzdir/$event/angle.txt
    #echo "$angle" >> $gzdir/$event/angle.txt
    event=$((event+1))

  done
done
