# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_opts_nortti_if_android")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")

#
# This is a TF Lite delegate that is used for on-device training exclusively.
#
package(
    default_visibility = [
        "//tensorflow/lite/delegates/flex:__subpackages__",
        "//third_party/fcp/client:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "training_delegate",
    srcs = [
        "training_delegate.cc",
    ],
    hdrs = [
        "training_delegate.h",
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    deps = [
        "//tensorflow/lite/delegates/flex:delegate_only_runtime",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
        ],
    }),
    alwayslink = 1,
)

tf_cc_test(
    name = "training_delegate_test",
    size = "small",
    srcs = [
        "training_delegate_test.cc",
    ],
    tags = [
        "no_gpu",  # GPU + flex is not supported.
    ],
    deps = [
        ":training_delegate",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/lite:shared_library",
        "//tensorflow/lite/delegates/flex:test_util",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)
