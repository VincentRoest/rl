#!/bin/bash

# check for input directory
if [ -z "$1" ]; then
    echo "ERROR: No input directory given"
    exit
else
    indir=$1
fi

# determine separator
if [ "$(uname)" == "Linux" ]; then
    connect='/'
else
    connect='\'
fi

if [ ! -d $indir ]; then
    echo "ERROR: Input directory '$indir' does not exist"
    exit
fi

for fname in $indir$connect*.pth; do
  [ -e "$fname" ] || continue
  echo $fname
  python remove_trainability.py $fname
done


