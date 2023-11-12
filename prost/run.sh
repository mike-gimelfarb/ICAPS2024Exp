#!/bin/bash
# usage: source run.sh <domain> <instance> <time>
CONTAINER="$1_$2_$3"
docker run --name $CONTAINER prost rddlprost $1 $2 $3
docker cp $CONTAINER:/OUTPUTS/. ./outputs
docker rm $CONTAINER