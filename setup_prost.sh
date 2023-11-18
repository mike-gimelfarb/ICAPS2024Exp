#!/bin/bash
# must run every time pyRDDLGym rddlsim.py global.cfg are changed
# use case: source ./setup_prost.sh <update pyRDDLGym>

# ensure workspace variables are set
echo "setting workspace variables"
export WORKSPACE=~/workspace
echo $WORKSPACE
export PROST_OUT="$PWD/outputs/prost"
echo $PROST_OUT

# reinstall pyRDDLGym
if [ "$1" = true ]; then
  echo "installing pyrddlgym"
  rm -rf $WORKSPACE/pyRDDLGym
  git clone https://github.com/ataitler/pyRDDLGym.git $WORKSPACE/pyRDDLGym
fi

# copy required run files for prost
echo "copying required files for prost"
cp rddlsim.py $WORKSPACE/pyRDDLGym/rddlsim.py
cp global.cfg $WORKSPACE/pyRDDLGym/global.cfg

