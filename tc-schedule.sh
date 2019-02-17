#!/bin/bash

set -ex

curdir=$(dirname "$0")

pip3 install --quiet --user --upgrade pip

curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/ef67832e6657f43e139a10f37eb326a7d9d96dad/requirements.txt | pip3 install --quiet --user --upgrade -r /dev/stdin
curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/ef67832e6657f43e139a10f37eb326a7d9d96dad/tc-decision.py > ${curdir}/tc-decision.py
python3 ${curdir}/tc-decision.py
