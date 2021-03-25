#!/bin/bash

SCRIPT=$(readlink "$0")
BASEDIR=$(dirname "$SCRIPT")
cd "$BASEDIR" || exit

if [ ! -d CodeCraft-2021 ]; then
  echo "ERROR: $BASEDIR is not a valid directory of SDK_python for CodeCraft-2021."
  echo "  Please run this script in a regular directory of SDK_python."
  exit 255
fi

TAG_NAME=$(git describe --tags)

rm -f "$TAG_NAME-"*.zip
zip -r "$TAG_NAME-$(date +%m-%d-%H:%M).zip" -i="CodeCraft-2021/*/[a-zA-Z]*" -x="*/__pycache__/*" CodeCraft-2021
