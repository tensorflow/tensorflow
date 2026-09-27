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

# For workaround https://github.com/bazelbuild/bazel/issues/8772 with Bazel >= 0.29.1
# TensorFlow still targets Java 1.7 (See JAVACOPTS in tensorflow/java/build_defs.bzl)
# which doesn't support "-parameters" flag. Starting from Java 11 (default since Bazel
# 0.29.1), a warning message will be thrown if "-parameters" is passed. If "-Werror" also exists,
# the compiling action will fail. To workaround this, we override the misc value of
# the default java toolchain to remove "-parameters" flag.
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain")

licenses(["notice"])

default_java_toolchain(
    name = "tf_java_toolchain",
    misc = [
        "-XDskipDuplicateBridges=true",
        "-g",
    ],
    visibility = ["//visibility:public"],
)
