package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "generate_op_registrations",
    srcs = ["gen_op_registration_main.cc"],
    deps = [
        "//tensorflow/contrib/lite/tools:gen_op_registration",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gen_op_registration",
    srcs = ["gen_op_registration.cc"],
    hdrs = ["gen_op_registration.h"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "gen_op_registration_test",
    srcs = ["gen_op_registration_test.cc"],
    data = [
        "//tensorflow/contrib/lite:testdata/0_subgraphs.bin",
        "//tensorflow/contrib/lite:testdata/2_subgraphs.bin",
        "//tensorflow/contrib/lite:testdata/empty_model.bin",
        "//tensorflow/contrib/lite:testdata/test_model.bin",
        "//tensorflow/contrib/lite:testdata/test_model_broken.bin",
    ],
    deps = [
        ":gen_op_registration",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mutable_op_resolver",
    srcs = ["mutable_op_resolver.cc"],
    hdrs = ["mutable_op_resolver.h"],
    deps = ["//tensorflow/contrib/lite:framework"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
