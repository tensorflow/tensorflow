# Fuzzing TensorFlow/core/function with GFT

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

tf_cc_fuzz_test(
    name = "runtime_client_fuzz",
    srcs = ["runtime_client_fuzz.cc"],
    deps = [
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/function/runtime_client:runtime_client_cc",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
    ],
)
