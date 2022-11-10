# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# A sample delegate that supports addition and subtraction only.
load("//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "sample_vendor_delegate",
    testonly = True,
    srcs = [
        "sample_vendor_delegate.cc",
    ],
    hdrs = [
        "sample_vendor_delegate.h",
    ],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "sample_vendor_delegate_external",
    testonly = True,
    srcs = [
        "sample_vendor_delegate_external.cc",
    ],
    copts = tflite_copts(),
    deps = [
        ":sample_vendor_delegate",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/experimental/acceleration/configuration/c:vendor_delegate",
        "//tensorflow/lite/core/shims:delegate_plugin",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/acceleration/configuration/c:vendor_delegate",
    ],
)

tflite_cc_shared_object(
    name = "tensorflowlite_sample_vendor",
    testonly = True,
    linkopts = tflite_linkopts(),
    deps = [
        ":sample_vendor_delegate_external",
    ],
)
