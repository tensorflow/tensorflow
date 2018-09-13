# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
FROM ubuntu:16.04

LABEL maintainer="Shanqing Cai <cais@google.com>"

# Copy and run the install scripts.
COPY install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN /install/install_deb_packages.sh

RUN apt-get update
RUN apt-get install -y --no-install-recommends python-pip
RUN pip install --upgrade wheel
RUN pip install --upgrade astor
RUN pip install --upgrade gast
RUN pip install --upgrade numpy
RUN pip install --upgrade termcolor
RUN pip install keras_applications==1.0.5
RUN pip install keras_preprocessing==1.0.3

# Install golang
RUN apt-get install -t xenial-backports -y golang-1.9
ENV PATH=${PATH}:/usr/lib/go-1.9/bin
