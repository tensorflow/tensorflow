# Test definitions for Lit, the LLVM test runner.
#
# This is reusing the LLVM Lit test runner in the interim until the new build
# rules are upstreamed.
# TODO(b/136126535): remove this custom rule.
"""Lit runner globbing test
"""

# This is a hack. It is just barely enough to work for our needs.
# Due to bazel's hermetic builds, if a test is not included in `data`,
# lit won't see it (and thus won't run it).
def _run_lit_test(name, data):
    """Runs lit on all tests it can find in `data` under tensorflow/compiler/mlir."""
    native.py_test(
        name = name,
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
        ],
        main = "lit.py",
    )

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
      per_test_extra_data: {str: [str]}, extra data to attach to a given file.
      default_tags: [str], additional tags to attach to the test.
      tags_override: {str: str}, tags to add to specific tests.
      driver: str, label of the driver shell script.
      features: [str], list of extra features to enable.
    """
    _run_lit_test("glob_lit_tests", data + native.glob(["*." + ext for ext in test_file_exts]))

def lit_test(
        name,
        data = [],
        size = "small",
        tags = None,
        driver = None,
        features = []):
    """Runs test files under lit.

    Args:
      name: str, the name of the test.
      data: [str], labels that should be provided as data inputs.
      size: str, the size of the test.
      tags: [str], tags to attach to the test.
      driver: str, label of the driver shell script.
      features: [str], list of extra features to enable.
    """
    _run_lit_test(name + ".test", data + [name])
