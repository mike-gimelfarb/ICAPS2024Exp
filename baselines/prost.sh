#!/bin/bash

# Confirm 4 parameters
if [ $# -ne 4 ]; then
    echo
    echo "  Usage: rddlprost <domain> <instance> <rounds> <time>"
    echo
    exit 1
fi

# File paths from rddlrepository
DOM="$1"
INST="$2"

echo "Starting RDDL Gym Server..."
( cd /workspace/pyRDDLGym && python prost.py "$DOM" "$INST" "$3" "$4" ) > pyrddl_$1_$2_prost_True_$4.log 2>&1 &

sleep 5
echo "Starting PROST..."
( cd /workspace/prost && ./search-release 813rocks \
"[PROST -s 1 -se [THTS -act [UCB1] -out [UMC] -backup [PB] -init [Expand -h [IDS]]]]" ) > prost_$1_$2_prost_True_$4.log 2>&1

echo "Writing data.json..."
cat /workspace/data.json > data_$1_$2_prost_True_$4.json
