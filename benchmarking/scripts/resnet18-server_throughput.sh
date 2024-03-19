#!/bin/bash

network="resnet18"
EXTRA="_throughput"
EXP_NAME="${network}${EXTRA}"
PROGRAM="/mnt/mohammad/delphi_bench/target/release/${network}-server"
OUTDIR="/mnt/mohammad/delphi_bench/benchmarking/outputs/${EXP_NAME}/server"
DATADIR="/mnt/mohammad/delphi_bench/benchmarking/data/${EXP_NAME}/server"
mkdir -p $OUTDIR
mkdir -p $DATADIR
BATCH=3



start=$(date +%s%N)
pids=()
for ((i=1; i<=BATCH; i++))
do
    echo "Starting server for batch $i"
    $PROGRAM "$i" "$EXP_NAME" > "${OUTDIR}/_batch_${i}.out" &
    pids+=($!)
done

# Wait for all processes to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

# End timing
end=$(date +%s%N)

# Calculate duration in milliseconds
duration=$(( (end - start) / 1000000 ))
echo "Total duration: $duration ms" > "${OUTDIR}/time_elapsed.txt"