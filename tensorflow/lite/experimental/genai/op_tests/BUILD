load("//tensorflow:strict.default.bzl", "py_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

py_strict_test(
    name = "transformer_test",
    srcs = ["transformer_test.py"],
    data = [
        "//tensorflow/lite/experimental/genai/op_tests/testdata:toy_model_prefill_decode.tflite",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/190842754): Enable test in OSS.
    ],
    deps = [
        "//testing/pybase",
        "@absl_py//absl/flags",
        #internal proto upb dep
        "//third_party/py/mock",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/lite/experimental/genai:genai_ops",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/lite/python:schema_util",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/lite/python/testdata:_pywrap_test_registerer",
        "//tensorflow/lite/tools:flatbuffer_utils",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
    ],
)
