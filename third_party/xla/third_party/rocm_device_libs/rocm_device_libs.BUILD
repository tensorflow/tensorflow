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

load("build_defs.bzl", "bitcode_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files([
    "LICENSE.TXT",
])

cc_binary(
    name = "prepare_builtins",
    srcs = glob([
        "utils/prepare-builtins/*.cpp",
        "utils/prepare-builtins/*.h",
    ]),
    copts = [
        "-fno-rtti -fno-exceptions",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:config",
    ],
)

bitcode_library(
    name = "ocml",
    srcs = glob([
        "ocml/src/*.cl",
    ]),
    hdrs = glob([
        "ocml/src/*.h",
        "ocml/inc/*.h",
        "irif/inc/*.h",
        "oclc/inc/*.h",
    ]),
    file_specific_flags = {
        "native_logF.cl": ["-fapprox-func"],
        "native_expF.cl": ["-fapprox-func"],
        "sqrtF.cl": ["-cl-fp32-correctly-rounded-divide-sqrt"],
    },
)

bitcode_library(
    name = "ockl",
    srcs = glob([
        "ockl/src/*.cl",
        "ockl/src/*.ll",
    ]),
    hdrs = glob([
        "ockl/inc/*.h",
        "irif/inc/*.h",
        "oclc/inc/*.h",
    ]),
    file_specific_flags = {
        "gaaf.cl": ["-munsafe-fp-atomics"],
    },
)
