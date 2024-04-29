#!/bin/bash

if [ "$#" -eq 1 ]; then
  output_file=$1
else
  # Default file name if no argument is provided
  output_file="memory_usage.csv"
fi

# Add CSV headers
echo "timestamp,total_memory,used_memory,free_memory" > "$output_file"

# Infinite loop to collect data every 10 seconds
while true; do
  # Read memory usage using 'free' and parse it with 'awk'
  read total used free <<< $(free -m | awk 'NR==2{print $2,$3,$4}')

  # Get the current timestamp
  timestamp=$(date +"%Y-%m-%d %H:%M:%S")

  # Write data to CSV
  echo "$timestamp,$total,$used,$free" >> "$output_file"

  # Wait for 10 seconds
  sleep 20
done