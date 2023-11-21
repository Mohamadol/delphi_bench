#!/bin/bash

# Program name or path
PROGRAM="/mnt/mohammad/delphi/rust/target/release/resnet50-client"

# Start each instance in the background
for i in {1..8}
do
   echo "Starting instance $i in the background"
   $PROGRAM $i &  # Start in the background
done

# Wait for all background jobs to finish
wait

echo