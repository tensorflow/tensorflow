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

# The rapids-logger project defines an easy way to produce a project-specific logger using the excellent spdlog package
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "rapids_logger",
    srcs = ["src/logger.cpp"],
    hdrs = glob(["include/rapids_logger/*.h*"]),
    copts = [
        "-std=c++17",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@spdlog",
    ],
)

cc_test(
    name = "smoke_test",
    srcs = ["smoke_test.cc"],
    deps = [
        ":rapids_logger",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
