load("//xla:py_strict.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_test(
    name = "tiled_kernel_test",
    srcs = ["tiled_kernel_test.py"],
    main = "tiled_kernel_test.py",
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "@absl_py//absl/testing:absltest",
    ],
)
