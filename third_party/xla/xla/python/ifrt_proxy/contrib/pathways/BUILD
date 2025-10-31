# Copyright 2025 The OpenXLA Authors.
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
load("//xla/python/ifrt_proxy/common:ifrt_proxy.bzl", "cc_library", "default_ifrt_proxy_visibility", "ifrt_proxy_cc_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_ifrt_proxy_visibility,
)

cc_library(
    name = "status_annotator_util",
    srcs = ["status_annotator_util.cc"],
    hdrs = ["status_annotator_util.h"],
    deps = [
        ":status_annotator_proto_cc",
        "//xla/python/ifrt:user_context",
        "//xla/python/ifrt:user_context_registry",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_to_from_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

ifrt_proxy_cc_test(
    name = "status_annotator_util_test",
    srcs = ["status_annotator_util_test.cc"],
    deps = [
        ":status_annotator_proto_cc",
        ":status_annotator_util",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt:user_context_registry",
        "//xla/python/ifrt:user_context_status_util",
        "//xla/tsl/platform:status_to_from_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "status_annotator_proto",
    srcs = ["status_annotator.proto"],
    visibility = default_ifrt_proxy_visibility,
    deps = ["//xla/tsl/protobuf:status_proto"],
)
