package(
    # To suppress build cleaner error about inclusion of schema_generate.h.
    features = ["-layering_check"],
)

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
        "//tensorflow/contrib/lite/toco:graph_transformations",
        "//tensorflow/contrib/lite/toco:model",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
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
        "@flatbuffers",
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
        "//tensorflow/contrib/lite:string_util",
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
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "export_test",
    srcs = [
        "export_test.cc",
    ],
    deps = [
        ":export",
        "//tensorflow/contrib/lite/schema:schema_fbs",
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
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/contrib/lite/toco:model",
        "//tensorflow/contrib/lite/toco:tooling_util",
        "//tensorflow/contrib/lite/tools:verifier",
        "@flatbuffers",
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
        "@flatbuffers",
    ],
)
