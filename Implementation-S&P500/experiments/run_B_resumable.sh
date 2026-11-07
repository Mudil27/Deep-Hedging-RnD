#!/usr/bin/env bash
# Keep relaunching Experiment B until all 300 windows are in the CSV.
# exp_B_30d.py skips windows that are already saved, so a crash or sleep only
# costs the window that was in progress.
cd "$(dirname "$0")/.."
TAG=${1:-_final}
CSV=results/exp_B${TAG}/windows.csv
for attempt in $(seq 1 50); do
    n=$( [ -f "$CSV" ] && echo $(( $(wc -l < "$CSV") - 1 )) || echo 0 )
    [ "$n" -ge 300 ] && { echo "all 300 windows done"; exit 0; }
    echo "=== attempt $attempt: $n windows done, (re)starting $(date)" >> results/exp_B${TAG}.log
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python experiments/exp_B_30d.py --tag "$TAG" >> results/exp_B${TAG}.log 2>&1
done
