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

# Description:
#   Implib.so is a simple equivalent of Windows DLL import libraries for POSIX
#   shared libraries.

load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

exports_files([
    "LICENSE.txt",
])

py_library(
    name = "implib_gen_lib",
    srcs = ["implib-gen.py"],
    data = glob([
        "arch/**/*.S.tpl",
        "arch/**/*.ini",
    ]),
)
