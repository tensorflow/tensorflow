# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""BUILD target helpers."""

load("//tensorflow:strict.default.bzl", "py_strict_test")

def reference_test(name, additional_deps = [], tags = [], shard_count = 1):
    py_strict_test(
        name = name,
        srcs = [name + ".py"],
        deps = [
            ":reference_tests",
            "//tensorflow:tensorflow_py_no_contrib",
        ] + additional_deps,
        python_version = "PY3",
        shard_count = shard_count,
        tags = tags + ["no_windows", "no_pip"],
    )
