#!/bin/bash

for file in runs/*.json; do
    python src/utils/evaluate.py "$file"
done
