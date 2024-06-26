#!/bin/bash

BATCH=4
MEMORY=30
CORES=8
network="resnet18"
EXTRA="_throughput_swap100"
EXP_NAME="${network}${EXTRA}"
PROGRAM="/mnt/mohammad/delphi_bench/target/release/${network}-server"
OUTDIR="/mnt/mohammad/delphi_bench/benchmarking/outputs/${EXP_NAME}/_${CORES}_${CORES}_${MEMORY}_${MEMORY}/_${BATCH}__batch/server"
DATADIR="/mnt/mohammad/delphi_bench/benchmarking/data/${EXP_NAME}/_${CORES}_${CORES}_${MEMORY}_${MEMORY}/_${BATCH}__batch/server"
mkdir -p $OUTDIR
mkdir -p $DATADIR


start=$(date +%s%N)
# pids=()
declare -a pids
cleanup() {
    echo "Cleaning up..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
}
trap cleanup SIGINT SIGTERM

for ((i=1; i<=BATCH; i++))
do
    echo "Starting server for batch $i"
    $PROGRAM "$i" "$EXP_NAME" "$CORES" "$MEMORY" "$BATCH" > "${OUTDIR}/_batch_${i}.out" &
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