load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

tf_cc_test(
    name = "op_handler_selector_test",
    srcs = ["op_handler_selector_test.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/tfrt/eager:core_runtime",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//backends/cpu:core_runtime",
    ],
)
