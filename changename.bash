#!/bin/bash

# Check if a directory path is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Change to the specified directory
cd "$1" || exit 1

# Rename the files
for file in *; do
    mv "$file" "hard_$file"
done

