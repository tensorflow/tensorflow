#!/bin/bash -x
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

mkdir ${DOWNLOADS_DIR}

EIGEN_HASH=62a2305d5734
if [ -f eigen.BUILD ]; then
	# Grab the current Eigen version name from the Bazel build file
	EIGEN_HASH=$(cat eigen.BUILD | grep archive_dir | head -1 | cut -f3 -d- | cut -f1 -d\")
fi

curl "https://bitbucket.org/eigen/eigen/get/${EIGEN_HASH}.tar.gz" \
-o /tmp/eigen-${EIGEN_HASH}.tar.gz
tar xzf /tmp/eigen-${EIGEN_HASH}.tar.gz -C ${DOWNLOADS_DIR}

git clone https://github.com/google/re2.git ${DOWNLOADS_DIR}/re2
git clone https://github.com/google/gemmlowp.git ${DOWNLOADS_DIR}/gemmlowp
git clone https://github.com/google/protobuf.git ${DOWNLOADS_DIR}/protobuf

# JPEG_VERSION=v9a
# curl "http://www.ijg.org/files/jpegsrc.${JPEG_VERSION}.tar.gz" \
# -o /tmp/jpegsrc.${JPEG_VERSION}.tar.gz
# tar xzf /tmp/jpegsrc.${JPEG_VERSION}.tar.gz -C ${DOWNLOADS_DIR}

# PNG_VERSION=v1.2.53
# curl -L "https://github.com/glennrp/libpng/archive/${PNG_VERSION}.zip" \
# -o /tmp/pngsrc.${PNG_VERSION}.zip
# unzip /tmp/pngsrc.${PNG_VERSION}.zip -d ${DOWNLOADS_DIR}
