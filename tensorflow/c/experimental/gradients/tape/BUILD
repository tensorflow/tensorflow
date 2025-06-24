# A tape built on top of unified execution APIs.
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tape_context",
    srcs = ["tape_context.cc"],
    hdrs = [
        "tape_context.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":tape_operation",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tape_operation",
    srcs = ["tape_operation.cc"],
    hdrs = [
        "tape_operation.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tape",
    hdrs = [
        "tape_context.h",
        "tape_operation.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":tape_context",
        ":tape_operation",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_function",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "tape_context.h",
        "tape_operation.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
)
