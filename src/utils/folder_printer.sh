#!/bin/bash

# Parent directory is the current directory
PARENT_DIR="."

# Loop through each directory under the current directory, excluding 'env' and '.ipynb_checkpoints'
find "$PARENT_DIR" -type d \( -name "env" -o -name ".ipynb_checkpoints" \) -prune -o -type d -print | while read -r dir; do
  # Count the number of files in the directory (ignoring subdirectories)
  file_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
  
  # Print the directory path and the file count
  echo "$dir: $file_count files"
done