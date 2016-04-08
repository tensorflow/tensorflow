#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script needs to be run from the tensorflow/tools/docs directory
# Pass -a to also rebuild C++ docs. This requires doxygen.

set -e

DOC_DIR="g3doc/api_docs"
DOXYGEN_BIN=${DOXYGEN:-doxygen}
DOXYGEN_CONFIG="tools/docs/tf-doxy_for_md-config"
# The TMP_DIR is set inside DOXYGEN_CONFIG and cannot be changed independently
TMP_DIR=/tmp/tensorflow-docs/xml

if [ ! -f gen_docs.sh ]; then
  echo "This script must be run from inside the tensorflow/tools/docs directory."
  exit 1
fi

# go to the tensorflow/ directory
pushd ../..
BASE=$(pwd)

# Make Python docs
bazel run -- //tensorflow/python:gen_docs_combined \
    --out_dir=$BASE/$DOC_DIR/python

# Check if we should build c++ docs (if -a is given)
if [ x$1 == x-a ]; then
  mkdir -p $TMP_DIR
  $DOXYGEN_BIN "$BASE/$DOXYGEN_CONFIG"
  bazel run -- //tensorflow/tools/docs:gen_cc_md \
      --out_dir=$BASE/$DOC_DIR/cc \
      --src_dir=$TMP_DIR
fi

popd
