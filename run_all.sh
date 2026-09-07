#!/bin/sh
# Rebuild every paper figure. Thin wrapper kept for muscle memory;
# the real entry point is make_figures.py.
exec python3 make_figures.py "$@"
