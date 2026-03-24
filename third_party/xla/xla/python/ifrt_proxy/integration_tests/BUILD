load("@rules_cc//cc:cc_library.bzl", "cc_library")

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

ifrt_proxy_cc_test(
    name = "client_impl_test_tfrt_cpu",
    srcs = ["client_impl_test_tfrt_cpu.cc"],
    deps = [
        ":scoped_pjrt_cpu_via_proxy",
        "//xla/python/ifrt:client_impl_test_lib",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "array_impl_test_tfrt_cpu",
    srcs = ["array_impl_test_tfrt_cpu.cc"],
    deps = [
        ":scoped_pjrt_cpu_via_proxy",
        "//xla/python/ifrt:array_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "array_impl_test_tfrt_cpu_with_grpc_large_transfer_optimization",
    srcs = ["array_impl_test_tfrt_cpu.cc"],
    args = [
        "--ifrt_proxy_grpc_large_transfer_optimization_threshold_bytes=0",
    ],
    env = {
        "IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY": "$$$$TEST_TMPDIR$$$$",
    },
    deps = [
        ":scoped_pjrt_cpu_via_proxy",
        "//xla/python/ifrt:array_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "executable_impl_test_tfrt_cpu",
    timeout = "moderate",
    srcs = ["executable_impl_test_tfrt_cpu.cc"],
    shuffle_tests = False,
    deps = [
        ":scoped_pjrt_cpu_via_proxy",
        "//xla/python/pjrt_ifrt:xla_executable_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "remap_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["remap_impl_test_tfrt_cpu.cc"],
    deps = [
        ":scoped_pjrt_cpu_via_proxy",
        "//xla/python/ifrt:remap_impl_test_lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

ifrt_proxy_cc_test(
    name = "mock_array_test",
    size = "small",
    srcs = ["mock_array_test.cc"],
    tags = [
        # Flakily triggers a double-free or related memory corruption after all
        # the tests are done (at exit time) in x86 Kokoro. We have not been able
        # to rootcause this after a few attempts (b/450109296).
        "no_oss",
    ],
    deps = [
        "//xla:future",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt_proxy/client",
        "//xla/python/ifrt_proxy/client:grpc_client",
        "//xla/python/ifrt_proxy/client:registry",
        "//xla/python/ifrt_proxy/server:grpc_server",
        "//xla/python/pjrt_ifrt",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "scoped_pjrt_cpu_via_proxy",
    testonly = True,
    srcs = ["scoped_pjrt_cpu_via_proxy.cc"],
    hdrs = ["scoped_pjrt_cpu_via_proxy.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt_proxy/client:grpc_client",
        "//xla/python/ifrt_proxy/client:registry",
        "//xla/python/ifrt_proxy/server:grpc_server",
        "//xla/python/pjrt_ifrt",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
