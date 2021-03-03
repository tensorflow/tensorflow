#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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


# Collection of helper functions that can be used in the different continuous
# integration scripts.

function die() {
  echo "$@" 1>&2 ; exit 1;
}

# A small utility to run the command and only print logs if the command fails.
# On success, all logs are hidden. This helps to keep the log output clean and
# makes debugging easier.
function readable_run {
  # Disable debug mode to avoid printing of variables here.
  set +x
  result=$("$@" 2>&1) || die "$result"
  echo "$@"
  echo "Command completed successfully at $(date)"
  set -x
}

# Check if the regex ${1} is to be found in the pathspec ${2}.
# An optional error messsage can be passed with ${3}
function check_contents() {
  GREP_OUTPUT=$(git grep -E -rn ${1} -- ${2})

  if [ "${GREP_OUTPUT}" ]; then
    echo "=============================================="
    echo "Found matches for ${1} that are not permitted."
    echo "${3}"
    echo "=============================================="
    echo "${GREP_OUTPUT}"
    return 1
  fi
}
