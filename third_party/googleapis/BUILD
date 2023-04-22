# Copyright 2020 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

cc_library(
    name = "grpc_all_cc",
    visibility = ["//tensorflow/core/platform/cloud:__pkg__"],
    deps = [
        "@com_google_googleapis//google/monitoring/v3:monitoring_cc_grpc",
    ],
)

cc_library(
    name = "protos_all_cc",
    visibility = ["//tensorflow/core/platform/cloud:__pkg__"],
    deps = [
        "@com_google_googleapis//google/monitoring/v3:monitoring_cc_proto_headers_only",
    ],
)

cc_library(
    name = "protos_all_cc_impl",
    visibility = ["//tensorflow/core/platform/cloud:__pkg__"],
    deps = [
        "@com_google_googleapis//google/monitoring/v3:monitoring_cc_proto",
    ],
)
