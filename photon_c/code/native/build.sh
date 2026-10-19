#!/bin/sh
# Build the PHOTON native kernel library.  The .so lands in the workspace and
# persists across conversation turns (unlike pip installs).
set -e
cd "$(dirname "$0")"
cc -O3 -march=native -fopenmp -fPIC -shared -Wall -ffp-contract=off -o libphoton_native.so photon_native.c
echo "built native/libphoton_native.so"
