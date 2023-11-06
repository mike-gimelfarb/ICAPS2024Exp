#!/bin/bash

# Confirm 3 parameters
if [ $# -ne 3 ]; then
    echo
    echo "  Usage: rddlprost <domain> <instance> <time>"
    echo
    exit 1
fi

# File paths from rddlrepository
echo "Starting RDDL Gym Server..."
( cd /workspace/pyRDDLGym && python prost.py "$1" "$2" 999999) > pyrddl_$1_$2_prost_True_$3.log 2>&1 &

sleep 5
echo "Starting PROST..."
( cd /workspace/prost && ./search-release 813rocks \
"[PROST -se [THTS -T TIME -t $3 -act [UCB1] -out [UMC] -backup [PB] -init [Expand -h [IDS]]]]" ) > prost_$1_$2_prost_True_$3.log 2>&1

echo "Writing json output file..."
cat /workspace/data.json > data_$1_$2_prost_True_$3.json
