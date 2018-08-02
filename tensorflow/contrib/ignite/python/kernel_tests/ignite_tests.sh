#!/usr/bin/env bash
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

set -e
set -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 start|stop <ignite container prefix>" >&2
  exit 1
fi

container=$2
if [ "$1" == "start" ]; then
    docker run -d --rm --net=host --volume $(pwd)/node1_entries:/entries/ --name=${container}_1 functorial/ignite-test
    docker run -d --rm --net=host --volume $(pwd)/node2_entries:/entries/ --name=${container}_2 functorial/ignite-test
    echo Wait 15 secs until ignite is up and running
    sleep 15
    echo Container ${container} started successfully
elif [ "$1" == "stop" ]; then
    docker rm -f $container_1
    echo Container ${container}_1 stopped successfully
    docker rm -f $container_2
    echo Container ${container}_2 stopped successfully
else
  echo "Usage: $0 start|stop <ignite container prefix>" >&2
  exit 1
fi