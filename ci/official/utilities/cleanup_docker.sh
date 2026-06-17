#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
cat <<EOF
IMPORTANT: These tests ran under docker. This script does not clean up the
container for you! You can delete the container with:

$ docker rm -f tf

You can also execute more commands within the container with e.g.:

$ docker exec tf bazel clean
$ docker exec -it tf bash
EOF

docker ps
