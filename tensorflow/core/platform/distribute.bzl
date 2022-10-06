"""Build rules for tf.distribute testing."""

load(
    "//devtools/build_cleaner/skylark:build_defs.bzl",
    "register_extension_info",
)
load(
    "//learning/brain/testing/tpu:local_multiworker.bzl",
    _local_2vm_tpu_test = "local_2vm_tpu_test",
    _local_multiworker_tpu_test = "local_multiworker_tpu_test",
)
load("//tensorflow/core/platform:distribute.oss.bzl", _distribute_py_test = "distribute_py_test")

def distribute_py_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        main = None,
        args = [],
        shard_count = 1,
        full_precision = False,
        xla_enable_strict_auto_jit = True,
        disable_mlir_bridge = True,
        disable_tpu_use_tfrt = None,
        **kwargs):
    """Generates py_test targets for CPU, GPU and TPU.

    Args:
        name: test target name to generate suffixed with `test`.
        srcs: source files for the tests.
        deps: additional dependencies for the test targets.
        tags: tags to be assigned to the different test targets.
        data: data files that need to be associated with the target files.
        main: main script to be run for the TPU test.
        args: arguments to the tests.
        shard_count: number of shards to split the tests across.
        full_precision: run TPU device in full precision mode.
        xla_enable_strict_auto_jit: whether to also run this with XLA.
        disable_mlir_bridge: whether to also run this with the mlir bridge enabled.
        disable_tpu_use_tfrt: None/bool. Whether generate a unit test on TFRT TPU
            Runtime.  If it is not specified, we automatically generate TFRT unit
            tests for targets managed by tensorflow team.
        **kwargs: extra keyword arguments to the non-tpu test.
    """

    tpu_args = (args + ["--xla_jf_conv_full_precision"] if full_precision else args)

    _distribute_py_test(
        name = name,
        srcs = srcs,
        deps = deps,
        tags = tags,
        data = data,
        main = main if main else "%s.py" % name,
        args = args,
        shard_count = shard_count,
        tpu_args = tpu_args,
        xla_enable_strict_auto_jit = xla_enable_strict_auto_jit,
        disable_mlir_bridge = disable_mlir_bridge,
        disable_tpu_use_tfrt = disable_tpu_use_tfrt,
        **kwargs
    )

    if ("notpu" not in tags and "no_tpu" not in tags and
        "nomultivm" not in tags):
        _local_2vm_tpu_test(
            name = "tpu_2vm_%s" % name,
            srcs = srcs,
            size = kwargs.get("size", "medium"),
            timeout = kwargs.get("timeout", "moderate"),
            tags = tags + ["tpu", "noguitar"] +
                   (["manual", "notap"] if shard_count > 1 else []),
            data = data,
            args = args,
            deps = deps,
        )

        _local_multiworker_tpu_test(
            name = "tpu_multiworker_%s" % name,
            srcs = srcs,
            size = "large",
            tags = tags + ["tpu", "noguitar", "manual", "notap"],
            data = data,
            args = args,
            deps = deps,
            disable_mlir_bridge = disable_mlir_bridge,
            disable_tfrt = disable_tpu_use_tfrt,
        )

register_extension_info(
    extension = distribute_py_test,
    label_regex_for_dep = "{extension_name}",
)
