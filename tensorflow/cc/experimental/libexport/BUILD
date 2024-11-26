load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility =
        [
            "//tensorflow/python/saved_model:__subpackages__",
        ],
    licenses = ["notice"],
)

cc_library(
    name = "save",
    srcs = ["save.cc"],
    hdrs = ["save.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "save_test",
    size = "small",
    srcs = ["save_test.cc"],
    deps = [
        ":save",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
    ],
)

cc_library(
    name = "load",
    srcs = ["load.cc"],
    hdrs = ["load.h"],
    deps = [
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/util/tensor_bundle",
        "//third_party/protobuf:protobuf_lite",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "load_test",
    size = "small",
    srcs = ["load_test.cc"],
    deps = [
        ":load",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/status:statusor",
    ],
)
