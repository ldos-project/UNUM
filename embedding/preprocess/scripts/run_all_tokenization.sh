#!/bin/bash

BOUNDARY_DIR="/datastor1/janec/datasets/boundaries"
SCRIPT="bucket_tokenize_mp.py"
NPROC=8
MAX_JOBS=4  # how many to run in parallel
job_count=0

for boundary_file in "$BOUNDARY_DIR"/*merged-relative.pkl; do
  echo "Processing $boundary_file"
  python3 "$SCRIPT" --boundaries-file "$boundary_file" --nprocs "$NPROC" &
  ((job_count++))

  # Wait if max concurrent jobs are running
  if ((job_count >= MAX_JOBS)); then
    wait -n
    ((job_count--))
  fi
done

# Wait for all remaining jobs to finish
wait
