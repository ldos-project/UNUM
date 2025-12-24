#!/bin/bash

SCRIPT="find_bucket_boundaries.py"

# Quantile boundaries
for B in 30 50 80 100; do
    echo "Running quantile with $B buckets..."
    python $SCRIPT --method quantile --num-buckets $B
done

# KMeans boundaries
for B in 30 50 80 100; do
    echo "Running kmeans with $B clusters..."
    python $SCRIPT --method kmeans --num-buckets $B
done

# Histogram (no num-buckets needed)
echo "Running histogram..."
python $SCRIPT --method histogram
