#!/bin/bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# Usage: get_test_list.sh OUTPUT BAZEL_TEST_COMMAND...
# Writes the list of tests that would be run from BAZEL_TEST_COMMAND to OUTPUT.
# Hides all extra output and always exits with success for now.

OUTPUT=$1
shift
"$@" --test_summary=short --check_tests_up_to_date 2>/dev/null | sort -u | awk '{print $1}' | grep "^//" | tee $OUTPUT
