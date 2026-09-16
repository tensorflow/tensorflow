# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Rules for generating unit-tests using hexagon delegates."""

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_hexagon_mobile_test")  #'@unused'

def hexagon_op_tests(
        srcs = [],
        deps = []):
    """Create both monolithic and individual unit test targets for each test file in 'srcs'.

    Args:
        srcs: list of test files, separate target will be created for each item in the list.
        deps: Dependencies will be added to all test targets.
    """

    for src in srcs:
        parts = src.split(".cc")
        cc_test(
            name = "hexagon_" + parts[0],
            srcs = [src],
            deps = deps,
            linkstatic = 1,
            tags = [
                "no_oss",
                "nobuilder",
                "notap",
            ],
        )

    all_ops_test_name = "hexagon_op_tests_all"
    cc_test(
        name = all_ops_test_name,
        srcs = srcs,
        deps = deps,
        linkstatic = 1,
        tags = [
            "no_oss",
            "nobuilder",
            "notap",
        ],
    )

    tflite_hexagon_mobile_test(all_ops_test_name)
