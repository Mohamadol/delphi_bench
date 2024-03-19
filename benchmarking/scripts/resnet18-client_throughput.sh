#!/bin/bash

network="resnet18"
EXTRA="_throughput"
EXP_NAME="${network}${EXTRA}"
PROGRAM="/mnt/mohammad/delphi_bench/target/release/${network}-client"
OUTDIR="/mnt/mohammad/delphi_bench/benchmarking/outputs/${EXP_NAME}/client"
DATADIR="/mnt/mohammad/delphi_bench/benchmarking/data/${EXP_NAME}/client"
mkdir -p $OUTDIR
mkdir -p $DATADIR
BATCH=8


start=$(date +%s%N)
pids=()
for i in {1..1}
do
    echo "Starting client for batch $i"
    $PROGRAM $i $EXP_NAME > "${OUTDIR}/_batch_${i}.out" 2>&1
done



for ((i=1; i<=BATCH; i++))
do
    cho "Starting client for batch $i"
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
echo "Total duration: $duration ms" > "${OUT_DIR}/time_elapsed.txt"