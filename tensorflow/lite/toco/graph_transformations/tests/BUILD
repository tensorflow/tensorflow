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

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "lstm_utils_test",
    srcs = ["lstm_utils_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "identify_l2_pool_test",
    srcs = ["identify_l2_pool_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "remove_successive_transpose_test",
    srcs = ["remove_successive_transpose_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "reorder_reshape_transpose_test",
    srcs = ["reorder_reshape_transpose_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_concatenation_test",
    srcs = ["resolve_constant_concatenation_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_pack_test",
    srcs = ["resolve_constant_pack_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "unpack_quantize_test",
    srcs = ["unpack_quantize_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_unary_test",
    srcs = ["resolve_constant_unary_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "identify_l2_normalization_test",
    srcs = ["identify_l2_normalization_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "fuse_binary_into_following_affine_test",
    srcs = ["fuse_binary_into_following_affine_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "fuse_binary_into_preceding_affine_test",
    srcs = ["fuse_binary_into_preceding_affine_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)
