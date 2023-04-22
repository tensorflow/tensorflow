#!/bin/bash
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
# Re-direct all links in $1 that point to /lib... to point to $1/lib... instead.

BASE="$1"
find "${BASE}" -type l | \
  while read l ; do
    if [[ "$(readlink "$l")" == /lib* ]]; then
      ORIG="$(readlink "$l")";
      rm "$l";
      ln -s "${BASE}${ORIG}" "$l"
    fi
  done

