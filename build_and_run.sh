#!/bin/bash

SCRIPT=$(readlink "$0")
BASEDIR=$(dirname "$SCRIPT")
cd "$BASEDIR" || exit

cd CodeCraft-2021 || exit

PY="python"
if [ "$(whereis pypy3 | wc -l)" -ne 0 ]; then
    PY="pypy3"
fi

if [ -n "$1" ]; then
    if [ -n "$2" ]; then
      $PY src/CodeCraft-2021.py "$1" "$2"
    else
      $PY src/CodeCraft-2021.py "$1"
    fi
else
    $PY src/CodeCraft-2021.py
fi
