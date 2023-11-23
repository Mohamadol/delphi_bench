#!/bin/bash

network="resnet50"
PROGRAM="/mnt/mohammad/delphi_bench/target/release/${network}-client"
OUTDIR="/mnt/mohammad/delphi_bench/benchmarking/outputs/${network}/client"
mkdir -p $OUTDIR

# Start each instance in the background
# for i in {1..8}
for i in {1..2}
do
   echo "Starting instance $i in the background"
    $PROGRAM $i > "${OUTDIR}/_batch_${i}.out" 2>&1
done

kill $pid

wait

echo