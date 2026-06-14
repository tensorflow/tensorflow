load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
)

licenses(["notice"])

py_test(
    name = "side_inputs_test",
    srcs = ["side_inputs_test.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "side_inputs_manual_api_test",
    srcs = ["side_inputs_manual_api_test.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
    ],
)
