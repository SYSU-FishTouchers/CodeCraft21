#!/bin/bash

SCRIPT=$(readlink "$0")
BASEDIR=$(dirname "$SCRIPT")
cd "$BASEDIR" || exit

cd CodeCraft-2021 || exit
python src/CodeCraft-2021.py