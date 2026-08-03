# Copyright 2026 The OpenXLA Authors.
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

load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//third_party/py/jax:internal",
        "//xla:friends",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = ["triton_add_kernel.hlo"],
)

xla_test(
    name = "triton_kernel_aot_test",
    srcs = ["triton_kernel_aot_test.cc"],
    backends = ["gpu"],
    data = ["triton_add_kernel.hlo"],
    tags = ["requires-gpu-nvidia"],
    deps = [
        "//third_party/py/jax/jaxlib/cuda:cuda_vendor",
        "//third_party/py/jax/jaxlib/cuda:triton_kernels",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/ffi",
        "//xla/ffi/api:c_api",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:path",
    ],
)
