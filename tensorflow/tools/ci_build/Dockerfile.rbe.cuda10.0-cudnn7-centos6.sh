#!/bin/bash -eux
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# Script to create a centos6 docker image.
# Before running, copy tensorrt into /tmp after downlading it from:
# https://developer.nvidia.com/nvidia-tensorrt-5x-download
#
# TODO(klimek): once there are downloadable images for tensorrt for centos6
# similar to debian, use those.
#
# Note that this creates a lot of large files in your filesystem:
# 1. The tensorrt rpm itself in your downloads folder.
# 2. The tensorrt copy in /tmp.
# 3. For each time this script is run and fails, there's a remaining copy
#    in the corresponding /tmp directory.
# 4. The docker cache grows large very quickly.
#
# To clean up failed script runs:
# $ rm /tmp/*/*.rpm

WORKDIR="$(mktemp -d)"
BASE="$(pwd)"
cp -R "${BASE}/"* "${WORKDIR}/"
cp "/tmp/nv-tensorrt-repo-rhel7-cuda10.0-trt5.1.5.0-ga-20190427-1-1.x86_64.rpm" "${WORKDIR}/tensorrt.rpm"
cd "${WORKDIR}"
docker build -f Dockerfile.rbe.cuda10.0-cudnn7-centos6 \
       --tag "gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-centos6" .
cd "${BASE}"
rm -rf "${WORKDIR}"
