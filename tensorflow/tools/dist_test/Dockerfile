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
# Docker image for testing distributed (GRPC) TensorFlow on Google Container
# Engine (GKE).
#
# See ./remote_test.sh for usage example.

FROM ubuntu:16.04

MAINTAINER Shanqing Cai <cais@google.com>

RUN apt-get update
RUN apt-get install -y \
    curl \
    python \
    python-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
RUN chmod +x install_google_cloud_sdk.bash
RUN ./install_google_cloud_sdk.bash --disable-prompts --install-dir=/var/gcloud

# Install kubectl
RUN /var/gcloud/google-cloud-sdk/bin/gcloud components install kubectl

# Install TensorFlow pip whl
# TODO(cais): Should we build it locally instead?
COPY tensorflow-*.whl /
RUN pip install /tensorflow-*.whl
RUN rm -f /tensorflow-*.whl

# Copy test files
COPY scripts /var/tf-dist-test/scripts
COPY python /var/tf-dist-test/python
