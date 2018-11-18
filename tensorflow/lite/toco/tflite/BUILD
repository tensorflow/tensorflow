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
        "whitelisted_flex_ops.cc",
        "whitelisted_flex_ops.h",
    ],
    hdrs = [
        "builtin_operator.h",
        "custom_operator.h",
        "operator.h",
        "simple_operator.h",
    ],
    deps = [
        ":types",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:ptr_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "operator_test",
    srcs = [
        "operator_test.cc",
    ],
    tags = ["no_oss"],
    deps = [
        ":operator",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:tooling_util",
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
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:model",
    ],
)

tf_cc_test(
    name = "types_test",
    srcs = [
        "types_test.cc",
    ],
    tags = ["no_oss"],
    deps = [
        ":types",
        "//tensorflow/core:ops",
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
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "//tensorflow/lite/tools/optimize:quantize_weights",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "export_test",
    srcs = [
        "export_test.cc",
    ],
    tags = ["no_oss"],
    deps = [
        ":export",
        "//tensorflow/core:ops",
        "//tensorflow/lite/schema:schema_fbs",
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
        "//tensorflow/lite:framework",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "//tensorflow/lite/tools:verifier",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "import_test",
    srcs = [
        "import_test.cc",
    ],
    tags = ["no_oss"],
    deps = [
        ":import",
        "//tensorflow/core:ops",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
