# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# TensorFlow Dockerfile Development Container
#
# You can use this image to quickly develop changes to the Dockerfile assembler
# or set of TF Docker partials. See README.md for usage instructions.
FROM ubuntu:16.04
LABEL maintainer="Austin Anderson <angerson@google.com>"

RUN apt-get update && apt-get install -y python3 python3-pip bash curl
RUN curl -sSL https://get.docker.com/ | sh
RUN pip3 install --upgrade pip setuptools pyyaml absl-py cerberus 'docker<=4.3.0'

WORKDIR /tf
VOLUME ["/tf"]

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
