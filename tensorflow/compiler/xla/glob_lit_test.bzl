# Test definitions for Lit, the LLVM test runner.
#
# This is reusing the LLVM Lit test runner in the interim until the new build
# rules are upstreamed.
# TODO(b/136126535): remove this custom rule.
"""Lit runner globbing test
"""

load("@bazel_skylib//lib:paths.bzl", "paths")

# Default values used by the test runner.
_default_test_file_exts = ["mlir", ".pbtxt", ".td"]
_default_driver = "@llvm-project//mlir:run_lit.sh"
_default_size = "small"
_default_tags = []

# These are patterns which we should never match, for tests, subdirectories, or
# test input data files.
_ALWAYS_EXCLUDE = [
    "**/LICENSE.txt",
    "**/README.txt",
    "**/lit.local.cfg",
    # Exclude input files that have spaces in their names, since bazel
    # cannot cope with such "targets" in the srcs list.
    "**/* *",
    "**/* */**",
]

def _run_lit_test(name, data, size, tags, driver, features, exec_properties):
    """Runs lit on all tests it can find in `data` under xla/.

    Note that, due to Bazel's hermetic builds, lit only sees the tests that
    are included in the `data` parameter, regardless of what other tests might
    exist in the directory searched.

    Args:
      name: str, the name of the test, including extension.
      data: [str], the data input to the test.
      size: str, the size of the test.
      tags: [str], tags to attach to the test.
      driver: str, label of the driver shell script.
              Note: use of a custom driver is not currently supported
              and specifying a default driver will abort the tests.
      features: [str], list of extra features to enable.
      exec_properties: may enable things like remote execution.
    """

    # Disable tests on windows for now, to enable testing rest of all xla and mlir.
    native.py_test(
        name = name,
        srcs = ["@llvm-project//llvm:lit"],
        tags = tags + ["no_pip", "no_windows"],
        args = [
            "xla/" + paths.basename(data[-1]) + " --config-prefix=runlit -v",
        ] + features,
        data = data + [
            "//tensorflow/compiler/xla:litfiles",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
        ],
        size = size,
        main = "lit.py",
        exec_properties = exec_properties,
    )

def glob_lit_tests(
        exclude = [],
        test_file_exts = _default_test_file_exts,
        default_size = _default_size,
        size_override = {},
        data = [],
        per_test_extra_data = {},
        default_tags = _default_tags,
        tags_override = {},
        driver = _default_driver,
        features = [],
        exec_properties = {}):
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
              Note: use of a custom driver is not currently supported
              and specifying a default driver will abort the tests.
      features: [str], list of extra features to enable.
      exec_properties: a dictionary of properties to pass on.
    """

    # Ignore some patterns by default for tests and input data.
    exclude = _ALWAYS_EXCLUDE + exclude

    tests = native.glob(
        ["*." + ext for ext in test_file_exts],
        exclude = exclude,
    )

    # Run tests individually such that errors can be attributed to a specific
    # failure.
    for curr_test in tests:
        # Instantiate this test with updated parameters.
        _run_lit_test(
            name = curr_test + ".test",
            data = data + [curr_test] + per_test_extra_data.get(curr_test, []),
            size = size_override.get(curr_test, default_size),
            tags = default_tags + tags_override.get(curr_test, []),
            driver = driver,
            features = features,
            exec_properties = exec_properties,
        )
