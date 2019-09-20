#!/bin/bash
#
# Copyright 2018 Google Inc. All rights reserved.
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
set -e

if [ -n "$1" ]; then
  scan_dir="$1"
else
  scan_dir="$( pwd )"
fi

py_checker="$0.py"

echo "scan root directory = '$scan_dir'"
python3 --version
# Scan recursively and search all *.cpp and *.h files using regex patterns.
# Assume that script running from a root of Flatbuffers working dir.
python3 $py_checker "ascii" "$scan_dir/include" "\.h$"
python3 $py_checker "ascii" "$scan_dir/src"     "\.cpp$"
python3 $py_checker "ascii" "$scan_dir/tests"   "\.h$"
python3 $py_checker "utf-8" "$scan_dir/tests"   "\.cpp$"
