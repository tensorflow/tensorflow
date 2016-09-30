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
#
# Docker image for testing distributed (GRPC) TensorFlow on a single machine.
#
# See ./local_test.sh for usage example.

FROM ubuntu:16.04

MAINTAINER Shanqing Cai <cais@google.com>

# Pick up some TF dependencies.
RUN apt-get update && apt-get install -y \
        python-numpy \
        python-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TensorFlow pip whl
# TODO(cais): Should we build it locally instead?
COPY tensorflow-*.whl /
RUN pip install /tensorflow-*.whl
RUN rm -f /tensorflow-*.whl

ADD . /var/tf_dist_test
