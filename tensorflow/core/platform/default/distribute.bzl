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
        tpu_tags = None,
        shard_count = 1,
        full_precision = False,
        disable_v2 = False,
        disable_v3 = False,
        disable_mlir_bridge = True,
        disable_tpu_use_tfrt = None,
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
        tpu_tags: tags for the tpu tests. If unspecified, uses value of `tags`.
        shard_count: number of shards to split the tests across.
        full_precision: unused.
        disable_v2: whether tests for TPU version 2 should be generated.
        disable_v3: whether tests for TPU version 3 should be generated.
        disable_mlir_bridge: whether to also run this with the mlir bridge enabled.
        disable_tpu_use_tfrt: None/bool. Whether generate a unit test on TFRT TPU
            Runtime.  If it is not specified, we automatically generate TFRT unit
            tests for targets managed by tensorflow team.
        **kwargs: extra keyword arguments to the non-tpu test.
    """

    # Default to PY3 since multi worker tests require PY3.
    kwargs.setdefault("python_version", "PY3")

    _ignore = (full_precision)
    tpu_tags = tags if (tpu_tags == None) else tpu_tags
    main = main if main else "%s.py" % name

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
            tags = tpu_tags,
            disable_v2 = disable_v2,
            disable_v3 = disable_v3,
            disable_mlir_bridge = disable_mlir_bridge,
            disable_tfrt = disable_tpu_use_tfrt,
        )
