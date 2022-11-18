# Description:
#   Tensor bundle: a module to efficiently serialize and deserialize tensors.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "if_windows",
    "tf_cc_test",
    "tf_copts",
)
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# To be exported to tensorflow/core:mobile_srcs.
filegroup(
    name = "mobile_srcs",
    srcs = [
        "byte_swap_array.h",
        "byte_swap_tensor.cc",
        "byte_swap_tensor.h",
        "naming.cc",
        "naming.h",
        "tensor_bundle.cc",
        "tensor_bundle.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
)

cc_library(
    name = "tensor_bundle",
    srcs = [
        "tensor_bundle.cc",
    ],
    hdrs = [
        "tensor_bundle.h",
    ],
    copts = tf_copts() + if_not_windows(["-Wno-sign-compare"]),
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":byteswaptensor",
        ":naming",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
    ],
)

cc_header_only_library(
    name = "tensor_bundle_headers_lib",
    features = ["-parse_headers"],  # Transitively pulls in Eigen headers
    deps = [":tensor_bundle"],
)

cc_library(
    name = "naming",
    srcs = ["naming.cc"],
    hdrs = ["naming.h"],
    deps = ["//tensorflow/core:lib"],
)

cc_library(
    name = "byteswaparray",
    hdrs = ["byte_swap_array.h"],
    deps = [
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/util:byte_swap_array",
    ],
)

cc_library(
    name = "byteswaptensor",
    srcs = ["byte_swap_tensor.cc"],
    hdrs = ["byte_swap_tensor.h"],
    deps = [
        ":byteswaparray",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:byte_order",
    ],
)

tf_cc_test(
    name = "tensor_bundle_test",
    srcs = ["tensor_bundle_test.cc"],
    data = ["//tensorflow/core/util/tensor_bundle/testdata:old_string_tensors"],
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        ":byteswaptensor",
        ":tensor_bundle",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)
