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

IGNITE_VERSION=2.6.0
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Start Apache Ignite with plain client listener.
docker run -itd --name ignite-plain -p 42300:10800 \
-v ${SCRIPT_PATH}:/data apacheignite/ignite:${IGNITE_VERSION} /data/bin/start-plain.sh

# Start Apache Ignite with IGFS.
docker run -itd --name ignite-igfs -p 10500:10500 \
-v ${SCRIPT_PATH}:/data apacheignite/ignite:${IGNITE_VERSION} /data/bin/start-igfs.sh