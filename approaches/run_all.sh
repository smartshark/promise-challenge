#!/bin/bash

py_approaches="baseline_rf_wp baseline_rf_all baseline_none baseline_all"

for py_approach in $py_approaches; do
    cd $py_approach
    source venv/bin/activate
    python approach.py ../../data ../../scores $py_approach 3 250
    deactivate
    cd ..
done
