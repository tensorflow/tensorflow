#!/usr/bin/env bash
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
# Install Hadoop

HADOOP_VERSION="2.7.3"

set +e
if [[ ! -f /usr/local/hadoop-${HADOOP_VERSION}/bin/hadoop ]]; then
  set -e
  wget -q http://www-us.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz
  tar xzf hadoop-${HADOOP_VERSION}.tar.gz -C /usr/local
fi
ln -sf /usr/local/hadoop-${HADOOP_VERSION} /usr/local/hadoop
