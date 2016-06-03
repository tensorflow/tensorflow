#!/bin/bash -x
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

set -e

DOWNLOADS_DIR=tensorflow/contrib/makefile/downloads

mkdir -p ${DOWNLOADS_DIR}

EIGEN_HASH=f3a13643ac1f
curl "https://bitbucket.org/eigen/eigen/get/${EIGEN_HASH}.tar.gz" \
-o /tmp/eigen-${EIGEN_HASH}.tar.gz
tar xzf /tmp/eigen-${EIGEN_HASH}.tar.gz -C ${DOWNLOADS_DIR}

clone_or_update_git_repo() {
  # Clone a git repo if local repo does not exist. Pull from master if it
  # does.
  # Usage: clone_or_update_git_repo <GIT_URL> <TARGET_DIR>
  GIT_URL=$1
  TARGET_DIR=$2

  TO_CLONE=0
  if [[ ! -d "${TARGET_DIR}" ]]; then
    TO_CLONE=1
  else
    pushd "${TARGET_DIR}" > /dev/null
    git rev-parse HEAD
    if [[ $? != "0" ]]; then
      TO_CLONE=1
    fi
    popd > /dev/null
  fi

  if [[ "${TO_CLONE}" == "1" ]]; then
    rm -rf "${TARGET_DIR}"
    git clone "${GIT_URL}" "${TARGET_DIR}"
  else
    pushd "${TARGET_DIR}" > /dev/null
    git pull origin master
    popd > /dev/null
  fi
}

clone_or_update_git_repo \
    https://github.com/google/re2.git ${DOWNLOADS_DIR}/re2
clone_or_update_git_repo \
    https://github.com/google/gemmlowp.git ${DOWNLOADS_DIR}/gemmlowp
clone_or_update_git_repo \
    https://github.com/google/protobuf.git ${DOWNLOADS_DIR}/protobuf

# JPEG_VERSION=v9a
# curl "http://www.ijg.org/files/jpegsrc.${JPEG_VERSION}.tar.gz" \
# -o /tmp/jpegsrc.${JPEG_VERSION}.tar.gz
# tar xzf /tmp/jpegsrc.${JPEG_VERSION}.tar.gz -C ${DOWNLOADS_DIR}

# PNG_VERSION=v1.2.53
# curl -L "https://github.com/glennrp/libpng/archive/${PNG_VERSION}.zip" \
# -o /tmp/pngsrc.${PNG_VERSION}.zip
# unzip /tmp/pngsrc.${PNG_VERSION}.zip -d ${DOWNLOADS_DIR}
