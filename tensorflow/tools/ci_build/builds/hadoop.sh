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

set -e

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export HADOOP_HDFS_HOME=/usr/local/hadoop
bazel test \
  --test_env=CLASSPATH=$($HADOOP_HDFS_HOME/bin/hadoop classpath --glob) \
  --test_env=HADOOP_HDFS_HOME=/usr/local/hadoop \
  --test_env=LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server \
  //tensorflow/core/platform/hadoop:hadoop_file_system_test
