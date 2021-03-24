#!/bin/bash

SCRIPT=$(readlink "$0")
BASEDIR=$(dirname "$SCRIPT")
cd "$BASEDIR" || exit

cd CodeCraft-2021 || exit

if [ -n "$1" ]; then
    if [ -n "$2" ]; then
      python src/CodeCraft-2021.py "$1" "$2"
    else
      python src/CodeCraft-2021.py "$1"
    fi
else
    python src/CodeCraft-2021.py
fi
