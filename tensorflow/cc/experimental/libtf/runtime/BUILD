load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/cc/experimental/libtf:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "runtime",
    srcs = [
        "runtime.cc",
    ],
    hdrs = [
        "runtime.h",
    ],
    deps = [
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:graph_function",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/cc/experimental/libexport:load",
        "//tensorflow/cc/experimental/libtf",
        "//tensorflow/cc/experimental/libtf:function",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
