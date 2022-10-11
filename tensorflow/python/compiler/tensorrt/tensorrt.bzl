"""Functions to initialize tensorrt python tests."""

load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")

TFTRT_PY_TEST_TAGS = [
    "no_cuda_on_cpu_tap",
    "no_rocm",
    "no_windows",
    "nomac",
]


TFTRT_PY_TEST_DEPS = [
    "//tensorflow/python:client_testlib",
    "//tensorflow/python:framework_test_lib",
    "//tensorflow/python/compiler/tensorrt:tf_trt_integration_test_base",
]

# buildifier: disable=unnamed-macro
def tftrt_py_tests(name, test_names, extra_tags=None, extra_deps=None, **kw):
    """A helper function that creates the Python TF-TRT unittests

    Args:
      name: str, the name of the group of tests.
      tags: a list of tags that will be used by each test.
      test_names: a list of tests OR a string representing the name of the test.
                  If test_names is an empty list, test_names = [name]
                  will be used.
    """

    if type(test_names) != "list":
        fail("The argument `test_names` received should be a list. " +
             "Received: `" + type(test_names) + "`.")
    elif not test_names:
        fail("The argument `test_names` is an empty list.")

    for test_name in test_names:
        tftrt_py_test(
            name=test_name,
            srcs=[test_name + ".py"],
            extra_tags=extra_tags,
            extra_deps=extra_deps,
            **kw
        )

# buildifier: disable=no-effect
def tftrt_py_test(*args, **kwargs):
    """Helper function providing a common base for Python TF-TRT Unittests."""

    kwargs.setdefault("python_version", "PY3")
    # cuda_py_test enable XLA tests by default. TensorRT can't combine XLA with
    # TensorRT currently and should set xla_enable_strict_auto_jit to False to
    # disable XLA tests.
    kwargs.setdefault("xla_enable_strict_auto_jit", False)
    kwargs.setdefault("tags", TFTRT_PY_TEST_TAGS)
    kwargs.setdefault("deps", TFTRT_PY_TEST_DEPS)
    kwargs.setdefault("extra_deps", [])
    kwargs.setdefault("extra_tags", [])

    for key in ["tags", "deps", "extra_deps", "extra_tags"]:
      kwargs[key] = [] if kwargs[key] == None else list(kwargs[key])

    kwargs["deps"] = kwargs["deps"] + kwargs.pop("extra_deps")
    kwargs["tags"] = kwargs["tags"] + kwargs.pop("extra_tags")

    cuda_py_test(*args, **kwargs)
