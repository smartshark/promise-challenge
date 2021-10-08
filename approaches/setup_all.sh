#!/bin/bash

py_approaches="baseline_rf_wp baseline_rf_all baseline_none baseline_all baseline_lr baseline_svm"

for py_approach in $py_approaches; do
    cd $py_approach
    echo $PWD
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    deactivate
    cd ..
done
