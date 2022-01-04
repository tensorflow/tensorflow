"""Generate regression test targets."""

load("//tensorflow:tensorflow.bzl", "py_strict_test")

_ALWAYS_EXCLUDE = ["*.disabled.mlir"]
_default_test_file_exts = ["mlir"]

def _run_regression_test(name, vectorize, data):
    suffix = ".vectorized.test" if vectorize else ".test"
    py_strict_test(
        name = name + suffix,
        srcs = ["compile_and_run_test.py"],
        args = [
            "--input_data_seed=1",
            "--test_file_name=" + name,
            "--vectorize=" + str(vectorize),
        ],
        data = data,
        python_version = "PY3",
        main = "compile_and_run_test.py",
        tags = [
            "no_pip",  # TODO(b/201803253): TFRT pybindings not in OSS.
            "nomsan",
        ],
        deps = [
            "@absl_py//absl/flags",
            "//third_party/py/mlir:ir",
            "//third_party/py/numpy",
            "//tensorflow/compiler/mlir/tfrt/jit/python_binding:tf_cpurt",
            "//tensorflow/python:client_testlib",
            "//tensorflow/python/platform",
        ],
    )

def regression_test(
        name,
        vectorize,
        exclude = [],
        test_file_exts = _default_test_file_exts,
        data = []):
    """ Generate regression tests.

    Args:
      name: The name of the test suite.
      vectorize: Whether vectorization should be enabled.
      exclude: The file patterns which should be excluded.
      test_file_exts: The file extensions to be considered as tests.
      data: Any extra data dependencies that might be needed.
    """
    exclude = _ALWAYS_EXCLUDE + exclude

    tests = native.glob(
        ["*." + ext for ext in test_file_exts],
        exclude = exclude,
    )

    for i in range(len(tests)):
        curr_test = tests[i]

        _run_regression_test(
            name = curr_test,
            vectorize = vectorize,
            data = data + [curr_test],
        )
