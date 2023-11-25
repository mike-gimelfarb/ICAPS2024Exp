#!/bin/bash

# Confirm 3 parameters
if [ $# -ne 3 ]; then
    echo
    echo "  Usage: prost.sh <domain> <instance> <time>"
    echo
    exit 1
fi

echo "Starting RDDL Gym Server..."
( python rddlsim.py "$1" "$2" "$3" ) > $PROST_OUT/rddlsim_$1_$2_prost_True_$3.log 2>&1 &
sleep 5

echo "Starting PROST..."
( cd $WORKSPACE/prost && ./prost.py $1_$2 \
"[PROST -se [THTS -T TIME -t $3 -act [UCB1] -out [UMC] -backup [PB] -init [Expand -h [IDS]]]]" ) > \
	$PROST_OUT/prost_$1_$2_prost_True_$3.log 2>&1
