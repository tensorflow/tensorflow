"""Build rules for tf.distribute testing."""

load("//tensorflow/python/tpu:tpu.bzl", _tpu_py_test = "tpu_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

def distribute_py_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        main = None,
        size = "medium",
        args = [],
        tpu_args = [],
        shard_count = 1,
        full_precision = False,
        **kwargs):
    """Generates py_test targets for CPU and GPU.

    Args:
        name: test target name to generate suffixed with `test`.
        srcs: source files for the tests.
        deps: additional dependencies for the test targets.
        tags: tags to be assigned to the different test targets.
        data: data files that need to be associated with the target files.
        main: optional main script.
        size: size of test, to control timeout.
        args: arguments to the non-tpu tests.
        tpu_args: arguments for the tpu tests.
        shard_count: number of shards to split the tests across.
        full_precision: unused.
        **kwargs: extra keyword arguments to the non-tpu test.
    """
    _ignore = (full_precision)

    cuda_py_test(
        name = name,
        srcs = srcs,
        data = data,
        main = main,
        size = size,
        deps = deps,
        shard_count = shard_count,
        tags = tags,
        args = args,
        **kwargs
    )

    if "notpu" not in tags and "no_tpu" not in tags:
        _tpu_py_test(
            disable_experimental = True,
            name = name + "_tpu",
            srcs = srcs,
            data = data,
            main = main,
            size = size,
            args = tpu_args,
            shard_count = shard_count,
            deps = deps,
            tags = tags,
        )
