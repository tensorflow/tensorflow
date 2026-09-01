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

%{multiline_comment}
cc_import(
    name = "curand_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcurand.so.%{libcurand_version}",
)
%{multiline_comment}
cc_library(
    name = "curand",
    %{comment}deps = [":curand_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/curand/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob(["include/curand*.h"]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
