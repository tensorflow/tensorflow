# Copyright 2023 The OpenXLA Authors.
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
load("//xla/python/ifrt_proxy/common:ifrt_proxy.bzl", "ifrt_proxy_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "register_pjrt_cpu_for_ifrt_api_tests",
    testonly = True,
    srcs = ["register_pjrt_cpu_for_ifrt_api_tests.cc"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt_proxy/client:grpc_client",
        "//xla/python/ifrt_proxy/client:registry",
        "//xla/python/ifrt_proxy/server:grpc_server",
        "//xla/python/pjrt_ifrt",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

ifrt_proxy_cc_test(
    name = "client_impl_test_tfrt_cpu",
    deps = [
        ":register_pjrt_cpu_for_ifrt_api_tests",
        "//xla/python/ifrt:client_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

ifrt_proxy_cc_test(
    name = "array_impl_test_tfrt_cpu",
    srcs = ["array_impl_test_tfrt_cpu.cc"],
    deps = [
        ":register_pjrt_cpu_for_ifrt_api_tests",
        "//xla/python/ifrt:array_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "executable_impl_test_tfrt_cpu",
    timeout = "moderate",
    srcs = ["executable_impl_test_tfrt_cpu.cc"],
    deps = [
        ":register_pjrt_cpu_for_ifrt_api_tests",  # buildcleaner: keep
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir/tests:executable_impl_test_lib",
        "//xla/python/pjrt_ifrt:xla_executable_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "remap_impl_test_tfrt_cpu",
    size = "small",
    srcs = [],
    deps = [
        ":register_pjrt_cpu_for_ifrt_api_tests",  # buildcleaner: keep
        "//xla/python/ifrt:remap_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

ifrt_proxy_cc_test(
    name = "mock_array_test",
    size = "small",
    srcs = ["mock_array_test.cc"],
    deps = [
        "//xla/pjrt/cpu:cpu_client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt_proxy/client",
        "//xla/python/ifrt_proxy/client:grpc_client",
        "//xla/python/ifrt_proxy/client:registry",
        "//xla/python/ifrt_proxy/server:grpc_server",
        "//xla/python/pjrt_ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)
