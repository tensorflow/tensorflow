load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Custom Ops useful for GenAI models.
load("//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "genai_ops",
    srcs = [
        "external_kvcache.cc",
        "genai_ops.cc",
        "kvcache.cc",
        "sdpa.cc",
    ],
    hdrs = [
        "genai_ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:subgraph",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/experimental/resource:cache_buffer",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:reference_base",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "kvcache_test",
    srcs = ["kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":genai_ops",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "external_kvcache_test",
    srcs = ["external_kvcache_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":genai_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "pywrap_genai_ops",
    srcs = [
        "genai_ops_wrapper.cc",
    ],
    hdrs = ["genai_ops.h"],
    additional_exported_symbols = ["GenAIOpsRegisterer"],
    common_lib_packages = [
        "litert/python",
        "tensorflow/lite/python",
    ],
    enable_stub_generation = True,
    link_in_framework = True,
    module_name = "pywrap_genai_ops",
    pytype_srcs = [
        "pywrap_genai_ops.pyi",
    ],
    visibility = ["//visibility:public"],
    wrap_py_init = True,
    deps = [
        ":genai_ops",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tflite_portable_test_suite()
