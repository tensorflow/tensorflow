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

licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "nccl_shared_library",
    shared_library = "lib/libnccl.so.%{libnccl_version}",
    hdrs = [":headers"],
    deps = ["@local_config_cuda//cuda:cuda_headers", ":headers"],
)
%{multiline_comment}
cc_library(
    name = "nccl",
    %{comment}deps = [":nccl_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/nccl/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/nccl*.h",
    %{comment}]),
    include_prefix = "third_party/nccl",
    includes = ["include/"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
