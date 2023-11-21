#!/bin/bash

network="resnet50"
PROGRAM="/mnt/mohammad/delphi/rust/target/release/${network}-client"
OUTDIR="/mnt/mohammad/delphi/rust/benchmarking/outputs/${network}/client"
mkdir -p $OUTDIR

# Start each instance in the background
for i in {1..8}
do
   echo "Starting instance $i in the background"
    $PROGRAM $i > "${OUTDIR}/_batch_${i}.out" 2>&1 &
done

# Wait for all background jobs to finish
wait

echo