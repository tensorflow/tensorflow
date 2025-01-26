"""Build rule for combining Tensorflow/XLA tests."""

load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")

def parse_label_name(label):
    """Parse a label into just the name.

    Args:
      label: string in relative or absolute form.

    Returns:
      The name of the label.
    """
    colon_split = label.split(":")
    if len(colon_split) == 1:  # no ":" in label
        return label
    return colon_split[-1]

def tf_xla_combined_py_test(name = "", package = None, tests = [], **kwargs):
    """Generates combined tf_xla_py_test targets, one per XLA backend.

    All srcs found in the list tests are combined into one new test which is then passed on to
    tf_xla_py_test which creates a new target per XLA backend.

    Args:
      name: Name of the target.
      package: The package that all tests in tests belong to.
      tests: The test targets to be combined and tested. Assumes all tests are in the same package.
      **kwargs: keyword arguments passed onto the tf_xla_py_test rule.
    """

    test_file = name + ".py"

    # run the generator to create the combined test file containing all the tests in test_files
    # redirecting the output of the generator to test_file.
    native.genrule(
        name = name + "_gen",
        testonly = 1,
        srcs = tests,
        outs = [test_file],
        cmd = """
mkdir -p $(@D) && cat > $@ << EOF
from tensorflow.python.platform import test
%s

if __name__ == "__main__":
  test.main()
EOF
        """ % "\n".join(["from %s.%s import *" % (package, parse_label_name(test)[:-4]) for test in tests]),
        tools = [],
        tags = ["generated_python_test=%s.%s" % (package, name)],
    )

    tf_xla_py_test(
        name = name,
        test_rule = py_strict_test,
        srcs = [test_file],
        deps = [
            "//tensorflow/python/platform:client_testlib",
        ] + tests,
        **kwargs
    )
