# Copyright 2026 Google Inc. All Rights Reserved.
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

"""Fuzzing template for TensorFlow ops."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def tf_ops_fuzz_target_lib(name):
    cc_library(
        name = name + "_fuzz_lib",
        srcs = [name + "_fuzz.cc"],
        deps = [
            "//tensorflow/core/kernels/fuzzing:fuzz_session",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/cc:ops",
            "//tensorflow/cc:scope",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//third_party/absl/log",
        ],
        tags = [
            "manual",
            "no_windows",
        ],
        alwayslink = 1,
        visibility = ["//visibility:public"],
    )

def tf_oss_fuzz_corpus(name):
    native.filegroup(
        name = name + "_corpus",
        srcs = native.glob(["corpus/" + name + "/*"]),
    )

def tf_oss_fuzz_dict(name):
    native.filegroup(
        name = name + "_dict",
        srcs = native.glob(["dictionaries/" + name + ".dict"]),
    )
