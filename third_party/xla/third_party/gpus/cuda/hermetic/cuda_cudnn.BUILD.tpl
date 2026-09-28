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
    name = "cudnn_ops_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_infer.so.%{libcudnn_ops_infer_version}",
)

cc_import(
    name = "cudnn_cnn_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_infer.so.%{libcudnn_cnn_infer_version}",
)

cc_import(
    name = "cudnn_ops_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_ops_train.so.%{libcudnn_ops_train_version}",
)

cc_import(
    name = "cudnn_cnn_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_cnn_train.so.%{libcudnn_cnn_train_version}",
)

cc_import(
    name = "cudnn_adv_infer",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_infer.so.%{libcudnn_adv_infer_version}",
)

cc_import(
    name = "cudnn_adv_train",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn_adv_train.so.%{libcudnn_adv_train_version}",
)

cc_import(
    name = "cudnn_main",
    hdrs = [":headers"],
    shared_library = "lib/libcudnn.so.%{libcudnn_version}",
)
%{multiline_comment}
cc_library(
    name = "cudnn",
    %{comment}deps = [
      %{comment}":cudnn_ops_infer",
      %{comment}":cudnn_ops_train",
      %{comment}":cudnn_cnn_infer",
      %{comment}":cudnn_cnn_train",
      %{comment}":cudnn_adv_infer",
      %{comment}":cudnn_adv_train",
      %{comment}"@cuda_nvrtc//:nvrtc",
      %{comment}":cudnn_main",
    %{comment}],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cudnn/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/cudnn*.h",
    %{comment}]),
    include_prefix = "third_party/gpus/cudnn",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
