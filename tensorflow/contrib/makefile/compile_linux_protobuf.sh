#!/bin/bash -e
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
# Builds protobuf 3 for Linux inside the local build tree.

GENDIR="$(pwd)/tensorflow/contrib/makefile/gen/protobuf"
HOST_GENDIR="$(pwd)/tensorflow/contrib/makefile/gen/protobuf-host"
mkdir -p "${GENDIR}"
ln -s "${GENDIR}" "${HOST_GENDIR}"

if [[ ! -f "tensorflow/contrib/makefile/downloads/protobuf/autogen.sh" ]]; then
    echo "You need to download dependencies before running this script." 1>&2
    echo "tensorflow/contrib/makefile/download_dependencies.sh" 1>&2
    exit 1
fi

cd tensorflow/contrib/makefile/downloads/protobuf

./autogen.sh
if [ $? -ne 0 ]
then
  echo "./autogen.sh command failed."
  exit 1
fi

./configure --prefix="${GENDIR}"
if [ $? -ne 0 ]
then
  echo "./configure command failed."
  exit 1
fi

make clean

make -j 8
if [ $? -ne 0 ]
then
  echo "make command failed."
  exit 1
fi

make install

echo "$(basename $0) finished successfully!!!"
