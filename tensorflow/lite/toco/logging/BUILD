load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "toco_conversion_log_proto",
    srcs = ["toco_conversion_log.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "conversion_log_util",
    srcs = ["conversion_log_util.cc"],
    hdrs = ["conversion_log_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":toco_conversion_log_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:version",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "//tensorflow/lite/toco/tflite:export",
        "//tensorflow/lite/toco/tflite:operator",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "conversion_log_util_test",
    srcs = ["conversion_log_util_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_log_util",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "html_template",
    srcs = [
        "template.html",
    ],
)

py_library(
    name = "gen_html",
    srcs = ["gen_html.py"],
    data = [
        "html_template",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "gen_html_test",
    srcs = ["gen_html_test.py"],
    data = [
        "//tensorflow/lite/toco/logging:template.html",
        "//tensorflow/lite/toco/logging/testdata:generated.html",
        "//tensorflow/lite/toco/logging/testdata:toco_log_after.pb",
        "//tensorflow/lite/toco/logging/testdata:toco_log_before.pb",
        "//tensorflow/lite/toco/logging/testdata:toco_tf_graph.dot",
        "//tensorflow/lite/toco/logging/testdata:toco_tflite_graph.dot",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gen_html",
        ":toco_conversion_log_proto_py",
        "//tensorflow/python:client_testlib",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "toco_conversion_log_proto_py",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":toco_conversion_log_proto"],
# )
# copybara:uncomment_end
