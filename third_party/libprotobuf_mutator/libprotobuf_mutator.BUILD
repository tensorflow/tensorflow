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

licenses(["notice"])  # Apache

exports_files(["LICENSE"])

cc_library(
    name = "libprotobuf_mutator_internals",
    srcs = [
        "src/binary_format.cc",
        "src/field_instance.h",
        "src/libfuzzer/libfuzzer_macro.cc",
        "src/libfuzzer/libfuzzer_mutator.cc",
        "src/mutator.cc",
        "src/text_format.cc",
        "src/utf8_fix.cc",
        "src/weighted_reservoir_sampler.h",
    ],
    hdrs = [
        "port/protobuf.h",
        "src/binary_format.h",
        "src/libfuzzer/libfuzzer_macro.h",
        "src/libfuzzer/libfuzzer_mutator.h",
        "src/mutator.h",
        "src/random.h",
        "src/text_format.h",
        "src/utf8_fix.h",
    ],
    includes = ["."],
    deps = ["@com_google_protobuf//:protobuf"],
)

cc_library(
    name = "libprotobuf_mutator",
    hdrs = ["src/libfuzzer/libfuzzer_macro.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [":libprotobuf_mutator_internals"],
)
