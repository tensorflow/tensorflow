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

# Fast C++ logging library. Header-only.
licenses(["notice"])  # MIT

exports_files(["LICENSE"])

cc_library(
    name = "spdlog",
    hdrs = glob(["include/spdlog/**/*.h"]),
    defines = [
        "SPDLOG_FMT_EXTERNAL",
    ],
    features = ["-parse_headers"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@fmt",
    ],
)

cc_test(
    name = "smoke_test",
    srcs = [
        "smoke_test.cc",  # lightweight test file
    ],
    copts = [
        "-DSPDLOG_FMT_EXTERNAL",
        "-fexceptions",
    ],
    deps = [
        ":spdlog",
        "@fmt",
    ],
)
