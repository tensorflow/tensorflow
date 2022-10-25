"""Functions to initialize tensorrt python tests."""

load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")

# buildifier: disable=unnamed-macro
def tensorrt_py_test(name, tags, test_names = []):
    """A helper function that creates the Python TF-TRT unittests

    Args:
      name: str, the name of the group of tests.
      tags: a list of tags that will be used by each test.
      test_names: a list of tests OR a string representing the name of the test.
                  If test_names is an empty list, test_names = [name]
                  will be used.
    """

    if type(test_names) != "list":
        if type(test_names) != "string":
            fail("Parameter 'test_names' of tensorrt_py_test(name = '" + name +
                 "', test_names = ...) should be a list or a string, got " +
                 "'" + type(test_names) + "'")
        test_names = [test_names]
    elif test_names == []:
        test_names = [name]

    for test_name in test_names:
        # cuda_py_test enable XLA tests by default. TensorRT can't combine XLA with
        # TensorRT currently and should set xla_enable_strict_auto_jit to False to
        # disable XLA tests.
        cuda_py_test(
            name = test_name,
            srcs = [test_name + ".py"],
            python_version = "PY3",
            tags = tags,
            xla_enable_strict_auto_jit = False,
            deps = [
                "//tensorflow/python:client_testlib",
                "//tensorflow/python:framework_test_lib",
                "//tensorflow/python/compiler/tensorrt:tf_trt_integration_test_base",
            ],
        )
