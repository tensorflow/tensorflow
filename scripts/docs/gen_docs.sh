#!/bin/bash
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This script needs to be run from the skflow/tools/docs directory

set -e

DOC_DIR="g3doc/api_docs"

if [ ! -f gen_docs.sh ]; then
  echo "This script must be run from inside the skflow/scripts/docs directory."
  exit 1
fi

# go to the skflow/ directory
pushd ../..
BASE=$(pwd)

# Make Python docs
python scripts/docs/gen_docs_combined.py --out_dir=$BASE/$DOC_DIR/python

popd
