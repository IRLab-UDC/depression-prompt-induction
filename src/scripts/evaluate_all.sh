#!/bin/bash

for file in runs/cross_domain/*.json; do
    python src/cross_domain_experiment/evaluate.py "$file"
done
