load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
)

licenses(["notice"])

py_strict_test(
    name = "side_inputs_test",
    srcs = ["side_inputs_test.py"],
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_test(
    name = "side_inputs_manual_api_test",
    srcs = ["side_inputs_manual_api_test.py"],
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
    ],
)
