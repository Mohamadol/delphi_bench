#!/bin/bash

network="resnet50"
PROGRAM="/mnt/mohammad/delphi_bench/target/release/${network}-client"
OUTDIR="/mnt/mohammad/delphi_bench/benchmarking/outputs/${network}/client"
DATADIR="/mnt/mohammad/delphi_bench/benchmarking/data/${network}/client"

mkdir -p $OUTDIR
mkdir -p $DATADIR

./memory_monitor.sh "${DATADIR}/memory_usage.csv" &
pid="$!"

# Start each instance in the background
for i in {1..1}
# for i in {1..2}
do
    echo "Starting client for batch $i"
    $PROGRAM $i > "${OUTDIR}/_batch_${i}.out" 2>&1
done

kill $pid

wait

echo