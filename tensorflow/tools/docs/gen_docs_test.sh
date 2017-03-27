#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

set -eux

if [ -d $TEST_SRCDIR/org_tensorflow ]; then
  TFDIR=$TEST_SRCDIR/org_tensorflow/tensorflow
else
  # Support 0.2.1- runfiles.
  TFDIR=$TEST_SRCDIR/tensorflow
fi
DOXYGEN=doxygen
DOXYGEN_CONFIG="tf-doxy_for_md-config"
TMP_DIR=/tmp/tensorflow-docs
mkdir -p $TMP_DIR/python
mkdir -p $TMP_DIR/xml
mkdir -p $TMP_DIR/cc

pushd $TFDIR
python/gen_docs_combined --out_dir=$TMP_DIR/python

# TODO(wicke): this does not work well inside the build/test jail
#$DOXYGEN "tools/docs/$DOXYGEN_CONFIG"
#tools/docs/gen_cc_md \
#    --out_dir=$TMP_DIR/cc \
#    --src_dir=$TMP_DIR/xml
popd
echo "PASS"
