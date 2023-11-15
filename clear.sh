#!/bin/bash
# to clear everything inside outputs folder and remake for all baselines
# use case: bash clear.sh
rm -rf outputs
mkdir outputs
cd outputs
mkdir jaxplan gurobiplan noop prost random
cd ..