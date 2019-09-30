# Test definitions for Lit, the LLVM test runner.
#
# This is reusing the LLVM Lit test runner in the interim until the new build
# rules are upstreamed.
# TODO(b/136126535): remove this custom rule.
"""Lit runner globbing test
"""

def glob_lit_tests(
        exclude = None,
        test_file_exts = ["mlir"],
        default_size = "small",
        size_override = None,
        data = None,
        per_test_extra_data = None,
        default_tags = None,
        tags_override = None,
        driver = None,
        features = []):
    """Creates all plausible Lit tests (and their inputs) under this directory.

    Args:
      exclude: [str], paths to exclude (for tests and inputs).
      test_file_exts: [str], extensions for files that are tests.
      default_size: str, the test size for targets not in "size_override".
      size_override: {str: str}, sizes to use for specific tests.
      data: [str], additional input data to the test.
      per_test_extra_data: {str: [str]}, extra data to attatch to a given file.
      default_tags: [str], additional tags to attach to the test.
      tags_override: {str: str}, tags to add to specific tests.
      driver: str, label of the driver shell script.
      features: [str], list of extra features to enable.
    """
    native.py_test(
        name = "glob_lit_tests",
        srcs = ["@llvm//:lit"],
        tags = ["no_rocm"],
        args = [
            "tensorflow/compiler/mlir --config-prefix=runlit",
        ],
        data = data + [
            "//tensorflow/compiler/mlir:litfiles",
            "@llvm//:FileCheck",
            "@llvm//:count",
            "@llvm//:not",
        ] + native.glob(["*." + ext for ext in test_file_exts]),
        main = "lit.py",
    )
