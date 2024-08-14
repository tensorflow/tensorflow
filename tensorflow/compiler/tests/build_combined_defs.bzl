"""Build rule for combining Tensorflow/XLA tests."""

load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

def tf_xla_combined_py_test(name = "", package = None, test_files = [], **kwargs):
    """Generates combined tf_xla_py_test targets, one per XLA backend.

    All tests found in the list test_files are combined into one new test which is then passed on to
    tf_xla_py_test which creates a new target per XLA backend.

    Args:
      name: Name of the target.
      package: The package that all tests in test_files belong to.
      test_files: The test files to be combined and tested.
      **kwargs: keyword arguments passed onto the tf_xla_py_test rule.
    """

    test_file = name + ".py"

    # run the generator to create the combined test file containing all the tests in test_files
    # redirecting the output of the generator to test_file.
    native.genrule(
        name = name + "_gen",
        testonly = 1,
        srcs = test_files,
        outs = [test_file],
        cmd = """
mkdir -p $(@D) && cat > $@ << EOF
from tensorflow.python.platform import test
%s

if __name__ == "__main__":
  test.main()
EOF
        """ % "\n".join(["from %s.%s import *" % (package, test[:-3]) for test in test_files]),
        tools = [],
        tags = ["generated_python_test=%s.%s" % (package, name)],
    )

    tf_xla_py_test(
        name = name,
        test_rule = py_strict_test,
        srcs = [test_file] + test_files,
        **kwargs
    )
