load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# Kernel for custom parse_example
package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite:string_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:feature_util",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "parse_example_test",
    srcs = ["parse_example_test.cc"],
    deps = [
        ":parse_example",
        "@flatbuffers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/example:feature_util",
            "//tensorflow/core/platform:protobuf",
            "//tensorflow/core/platform:tstring",
        ],
    }),
)
