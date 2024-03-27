# Copyright 2024 The OpenXLA Authors.
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
# ============================================================================

# Placeholder: load py_test
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//xla:pytype.default.bzl", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "configure",
    srcs = ["configure.py"],
)

py_test(
    name = "configure_test",
    srcs = ["configure_test.py"],
    data = [
        "testdata/clang.bazelrc",
        "testdata/cuda_clang.bazelrc",
        "testdata/gcc.bazelrc",
        "testdata/nvcc_clang.bazelrc",
        "testdata/nvcc_gcc.bazelrc",
    ],
    deps = [
        ":configure",
        "//build_tools:test_utils",
        "@absl_py//absl/testing:absltest",
    ],
)

# Below targets are just for checking if the host/CUDA compiler are configured
# as expected.
cc_library(
    name = "assert_clang",
    srcs = ["assert_clang.cc"],
    tags = ["manual"],
)

cc_library(
    name = "assert_gcc",
    srcs = ["assert_gcc.cc"],
    tags = ["manual"],
)

cuda_library(
    name = "assert_cuda_clang",
    srcs = ["assert_cuda_clang.cu.cc"],
    tags = [
        "gpu",
        "manual",
    ],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

cuda_library(
    name = "assert_nvcc",
    srcs = ["assert_nvcc.cu.cc"],
    tags = [
        "gpu",
        "manual",
    ],
    # Notably, this builds fine in OSS without this dependency. Apparently,
    # NVCC can give targets access to CUDA headers without letting Bazel know,
    # while CUDA clang cannot.
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
