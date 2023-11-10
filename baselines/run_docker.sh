#!/bin/bash
CONTAINER="$1-$2-$3"
docker run --name $CONTAINER prost rddlprost $1 $2 $3
docker cp $CONTAINER:/OUTPUTS/. ../outputs/prost
docker rm $CONTAINER