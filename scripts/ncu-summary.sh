#!/usr/bin/env bash
# Dump the docs/reduce.md Speed-Of-Light columns for one or more ncu reports.
# The version label comes from the file name (v1.ncu-rep -> v1); N and Comment
# are left blank for you to fill. Filters to the "GPU Speed Of Light Throughput"
# section so the %-valued Memory Throughput is picked (not the Gbyte/s one).
#
# Run inside the pixi env so `ncu` is on PATH, e.g.:
#   pixi run ncu-row v0.ncu-rep v1.ncu-rep
set -euo pipefail

printf '| Version | N | Duration (us) | Compute %% | Memory %% | L1/TEX %% | DRAM %% | Comment |\n'
printf '| --- | --- | --- | --- | --- | --- | --- | --- |\n'
for rep in "$@"; do
  ver=$(basename "$rep" .ncu-rep)
  ncu --import "$rep" --csv --page details 2>/dev/null | awk -F'","' -v ver="$ver" '
    $12 == "GPU Speed Of Light Throughput" { val = $15; sub(/",?$/, "", val); m[$13] = val }
    END { printf "| %s | | %s | %s | %s | %s | %s | |\n", ver, m["Duration"], m["Compute (SM) Throughput"], m["Memory Throughput"], m["L1/TEX Cache Throughput"], m["DRAM Throughput"] }'
done
