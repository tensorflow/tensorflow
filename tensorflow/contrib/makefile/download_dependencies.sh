#!/bin/bash -ex
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

DOWNLOADS_DIR=tensorflow/contrib/makefile/downloads
BZL_FILE_PATH=tensorflow/workspace.bzl

mkdir -p ${DOWNLOADS_DIR}

# Grab the current Eigen version name from the Bazel build file
EIGEN_HASH=$(cat "${BZL_FILE_PATH}" | egrep "eigen_version.*=.*\".*\"" | awk '{ print $3 }')
# Trim trailing and preceding double quotes
EIGEN_HASH="${EIGEN_HASH%\"}"
EIGEN_HASH="${EIGEN_HASH#\"}"

if [[ -z "${EIGEN_HASH}" ]]; then
    echo >&2 "Eigen hash does not exist."
    exit 1
else
    echo "Eigen hash = ${EIGEN_HASH}"
fi

curl "https://bitbucket.org/eigen/eigen/get/${EIGEN_HASH}.tar.gz" \
-o /tmp/eigen-${EIGEN_HASH}.tar.gz
tar xzf /tmp/eigen-${EIGEN_HASH}.tar.gz -C ${DOWNLOADS_DIR}

git clone https://github.com/google/re2.git ${DOWNLOADS_DIR}/re2
git clone https://github.com/google/gemmlowp.git ${DOWNLOADS_DIR}/gemmlowp
git clone https://github.com/google/protobuf.git ${DOWNLOADS_DIR}/protobuf

# Link to the downloaded Eigen library from a permanent directory name, since
# the downloaded name changes with every version.
cd ${DOWNLOADS_DIR}
rm -rf eigen-latest
ln -s eigen-eigen-${EIGEN_HASH} eigen-latest

echo "download_dependencies.sh completed successfully."
