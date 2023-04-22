#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# This script renames the shared objects with the nightly date from that the date
# you invoke the script. You can run this on Linux or MacOS.

# Examples Before and After (Linux)
# Before
# libtensorflow.so.2.3.0
# libtensorflow_framework.so.2.3.0
# libtensorflow_framework.so
# libtensorflow_framework.so.2
# libtensorflow.so
# libtensorflow.so.2

# After
# libtf_nightly_framework.so
# libtf_nightly_framework.so.06102020
# libtf_nightly.so
# libtf_nightly.so.06102020

DATE=$(date +'%m%d%Y')

# Get path to lib directory containing all shared objects.
if [[ -z "$1" ]]; then
  echo
  echo "ERROR: Please provide a path to the extracted directory named lib containing all the shared objects."
  exit 1
else
  DIRNAME=$1
fi

# Check if this
if test -f "${DIRNAME}/libtensorflow_framework.so"; then
  FILE_EXTENSION=".so"
elif test -f "${DIRNAME}/libtensorflow_framework.dylib"; then
  FILE_EXTENSION=".dylib"
else
  echo
  echo "ERROR: The directory provided did not contain a .so or .dylib file."
  exit 1
fi

pushd ${DIRNAME}

# Remove currently symlinks.
# MacOS
if [ $FILE_EXTENSION == ".dylib" ]; then
  rm -rf *.2${FILE_EXTENSION}
  rm -rf libtensorflow${FILE_EXTENSION}
  rm -rf libtensorflow_framework${FILE_EXTENSION}
# Linux
else
  rm -rf *${FILE_EXTENSION}
  rm -rf *${FILE_EXTENSION}.2
fi


# Rename the shared objects and symlink.
# MacOS
if [ $FILE_EXTENSION == ".dylib" ]; then
  # Rename libtensorflow_framework to libtf_nightly_framework.
  mv libtensorflow_framework.*${FILE_EXTENSION} libtf_nightly_framework.${DATE}${FILE_EXTENSION}
  ln -s libtf_nightly_framework.${DATE}${FILE_EXTENSION} libtf_nightly_framework${FILE_EXTENSION}

  # Rename libtensorflow to libtf_nightly.
  mv libtensorflow.*${FILE_EXTENSION} libtf_nightly.${DATE}${FILE_EXTENSION}
  ln -s libtf_nightly.${DATE}${FILE_EXTENSION} libtf_nightly${FILE_EXTENSION}
# Linux
else
  # Rename libtensorflow_framework to libtf_nightly_framework.
  mv libtensorflow_framework${FILE_EXTENSION}.* libtf_nightly_framework${FILE_EXTENSION}.${DATE}
  ln -s libtf_nightly_framework${FILE_EXTENSION}.${DATE} libtf_nightly_framework${FILE_EXTENSION}

  # Rename libtensorflow to libtf_nightly.
  mv libtensorflow${FILE_EXTENSION}.* libtf_nightly${FILE_EXTENSION}.${DATE}
  ln -s libtf_nightly${FILE_EXTENSION}.${DATE} libtf_nightly${FILE_EXTENSION}
fi


echo "Successfully renamed the shared objects with the tf-nightly format."

