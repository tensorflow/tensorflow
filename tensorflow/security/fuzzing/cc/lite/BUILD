# Fuzzers for TensorFlow Lite.
#
# These targets fuzz the TFLite Interpreter load/prepare/invoke pipeline,
# which is exercised by tf.lite.Interpreter(model_content=...) from Python.

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "tflite_interpreter_fuzz",
    srcs = ["tflite_interpreter_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)
