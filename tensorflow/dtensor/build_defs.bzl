"""Helpers for defining multi-platform DTensor test targets."""

load("//tensorflow:tensorflow.bzl", "py_strict_test")

# LINT.IfChange
ALL_BACKENDS = ["cpu", "gpu", "tpu"]
TPU_V3_DONUT_BACKEND = "tpu_v3_2x2"
TPU_V4_DONUT_BACKEND = "tpu_v4_2x2"
GPU_2DEVS_BACKEND = "2gpus"
PATHWAYS = "pw"
# LINT.ThenChange(
#     python/tests/test_backend_name.py:backend_name,
#     python/tests/test_backend_name.oss.py:backend_name
# )

# FIXME(feyu): Gradually increase the coverage of OSS tests.
def _get_configurations(
        disable,
        enable,
        disable_tfrt,
        backend_tags,
        backend_deps,
        additional_backends,  # buildifier: disable=unused-variable
        shard_count):
    """Returns a list of dtensor_test configurations."""
    disabled_tags = ["manual", "disabled"]
    disabled_tfrt_configs = [d + "_tfrt" for d in disable_tfrt]
    disabled_backends = [backend for backend in disable if backend not in enable]
    configurations = [
        dict(suffix = "cpu", backend = "cpu", tags = [], flags = [], env = {}, deps = []),
    ]

    backend_variant_deps = {}

    # Post processing configurations.
    for config in configurations:
        config["tags"] = config["tags"] + backend_tags.get(config["backend"], [])
        config["env"]["DTENSOR_TEST_UTIL_BACKEND"] = config["suffix"]

        if config["backend"] in disabled_backends or config["suffix"] in disabled_tfrt_configs:
            config["tags"] += disabled_tags

        config["deps"] = (
            config["deps"] +
            backend_variant_deps.get(config["backend"], []) +
            backend_deps.get(config["backend"], [])
        )
        config["shard_count"] = shard_count.get(config["backend"], None) if shard_count else None
    return configurations

def dtensor_test(
        name,
        srcs,
        deps = [],
        args = [],
        env = {},
        disable = [],
        enable = [],
        disable_tfrt = [],
        data = [],
        tags = [],
        backend_tags = {},
        backend_deps = {},
        additional_backends = [],
        main = None,
        shard_count = None,
        size = None,
        get_configurations = _get_configurations):
    """Defines a set of per-platform DTensor test targets.

    Generates test targets named:
    :name  # test suite that tests all backends
    :name_cpu
    :name_cpu_tfrt
    :name_gpu  # must run with --config=cuda
    :name_tpu  # recommend to be run with -c opt
    :name_tpu_tfrt  # recommend to be run with -c opt

    Args:
      name: test name
      srcs: source files
      deps: dependencies
      args: arguments to pass to the test
      env: environment variables to set when the test is executed
      disable: list of backends on which the test should be disabled, e.g., ["cpu"]
      enable: list of specific configs on which the test should be enabled,
        e.g., ["tpu"]. This overrides 'disable'.
      disable_tfrt: list of backends that are disabled for tfrt. This overrides 'enable'.
      data: data dependencies
      tags: test tags
      backend_tags: a dictionary keyed by backend name of per-backend tags.
      backend_deps: a dictionary keyed by backend name of per-backend deps.
      additional_backends: list of backends in addition to common cpu/tpu/gpu.
      main: the Python main file.
      shard_count: a dictionary keyed by backend name of per-backend shard counts.
      size: the test size.
      get_configurations: a function that returns the list of configurations. Used to generate non-OSS test targets.
    """
    configurations = get_configurations(
        disable = disable,
        enable = enable,
        disable_tfrt = disable_tfrt,
        backend_tags = backend_tags,
        backend_deps = backend_deps,
        additional_backends = additional_backends,
        shard_count = shard_count,
    )

    if main == None:
        if len(srcs) == 1:
            main = srcs[0]
        else:
            fail("Only one test source file is currently supported.")

    all_tests = []
    for config in configurations:
        config_name = name + "_" + config["suffix"]

        all_tests.append(config_name)

        test_rule = py_strict_test
        python_version = "PY3"
        test_env = {}
        test_env.update(config["env"])
        test_env.update(env)

        test_rule(
            env = test_env,
            name = config_name,
            main = main,
            srcs = srcs,
            data = data,
            args = config["flags"] + args,
            deps = config["deps"] + deps,
            tags = config["tags"] + tags,
            python_version = python_version,
            shard_count = config["shard_count"],
            size = size,
        )
    native.test_suite(name = name, tests = all_tests, tags = ["-manual"])
