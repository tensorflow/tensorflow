#include "third_party/absl/strings/str_cat.h"
#Description:
# TensorFlow cc tools.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "freeze_saved_model",
    srcs = ["freeze_saved_model.cc"],
    hdrs = ["freeze_saved_model.h"],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "freeze_saved_model_test",
    srcs = ["freeze_saved_model_test.cc"],
    deps = [
        ":freeze_saved_model",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
