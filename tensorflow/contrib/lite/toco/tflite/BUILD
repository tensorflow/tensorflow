licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

cc_library(
    name = "operator",
    srcs = [
        "operator.cc",
    ],
    hdrs = [
        "builtin_operator.h",
        "custom_operator.h",
        "operator.h",
        "simple_operator.h",
    ],
    deps = [
        ":types",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/contrib/lite/toco:model",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@flatbuffers//:flatbuffers",
    ],
)

tf_cc_test(
    name = "operator_test",
    srcs = [
        "operator_test.cc",
    ],
    deps = [
        ":operator",
        "//tensorflow/contrib/lite/toco:tooling_util",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:flatbuffers",
    ],
)

cc_library(
    name = "types",
    srcs = [
        "types.cc",
    ],
    hdrs = [
        "types.h",
    ],
    deps = [
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/contrib/lite/toco:model",
    ],
)

tf_cc_test(
    name = "types_test",
    srcs = [
        "types_test.cc",
    ],
    deps = [
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "export",
    srcs = [
        "export.cc",
    ],
    hdrs = [
        "export.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":operator",
        ":types",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/contrib/lite/toco:model",
        "//tensorflow/contrib/lite/toco:tooling_util",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:flatbuffers",
    ],
)

tf_cc_test(
    name = "export_test",
    srcs = [
        "export_test.cc",
    ],
    deps = [
        ":export",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "import",
    srcs = [
        "import.cc",
    ],
    hdrs = [
        "import.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":operator",
        ":types",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/contrib/lite/toco:model",
        "@flatbuffers//:flatbuffers",
    ],
)

tf_cc_test(
    name = "import_test",
    srcs = [
        "import_test.cc",
    ],
    deps = [
        ":import",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:flatbuffers",
    ],
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
