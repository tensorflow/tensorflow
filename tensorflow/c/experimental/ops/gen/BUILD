load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "generate_cpp_main",
    srcs = ["generate_cpp_main.cc"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/c/experimental/ops/gen/cpp",
        "//tensorflow/c/experimental/ops/gen/model",
        "//tensorflow/core:lib",
        # Without this line to link in ops, the global registry will be empty:
        "//tensorflow/core:ops",
    ],
)

tf_cc_binary(
    name = "generate_cpp",
    args = ["--stderrthreshold=1"],  # Warnings and errors to stderr.
    deps = [":generate_cpp_main"],
)
