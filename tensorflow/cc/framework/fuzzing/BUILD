# TODO(unda): describe this package.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(
    "//tensorflow/cc/framework/fuzzing:op_fuzzing.bzl",
    "tf_gen_op_wrappers_fuzz",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "cc_op_fuzz_gen_main",
    srcs = [
        "cc_op_fuzz_gen.cc",
        "cc_op_fuzz_gen.h",
        "cc_op_fuzz_gen_main.cc",
    ],
    copts = tf_copts(),
    data = [
        "//tensorflow/core/api_def:base_api_def",
    ],
    deps = [
        "//tensorflow/cc:cc_op_gen_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

# copybara:uncomment_begin
# tf_gen_op_wrappers_fuzz(
#     name = "array_ops_fuzz",
#     api_def_srcs = ["//tensorflow/core/api_def:base_api_def"],
#     extra_gen_deps = ["//tensorflow/c/kernels:bitcast_op_lib"],
#     op_lib_names = [
#         "array_ops",
#     ],
#     pkg = "//tensorflow/core",
#     deps = [
#         "//tensorflow/cc:cc_ops",
#         "//tensorflow/core/kernels:array",
#     ],
# )
# copybara:uncomment_end

bzl_library(
    name = "op_fuzzing_bzl",
    srcs = ["op_fuzzing.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow:tensorflow_bzl",
        "//tensorflow/core/platform:rules_cc_bzl",
    ],
)
