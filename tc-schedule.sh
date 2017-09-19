#!/bin/bash

set -ex

curdir=$(dirname "$0")

pip3 install --quiet --user --upgrade pip

curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/0ee93e9f36ef1fc77218a96e865daa80d36b87e5/requirements.txt | pip3 install --quiet --user --upgrade -r /dev/stdin
curl -L --silent https://raw.githubusercontent.com/lissyx/taskcluster-github-decision/0ee93e9f36ef1fc77218a96e865daa80d36b87e5/tc-decision.py > ${curdir}/tc-decision.py
python3 ${curdir}/tc-decision.py
