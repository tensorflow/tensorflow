"""Build rules for Tensorflow/XLA testing."""

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/compiler/tests:plugin.bzl", "plugins")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)

all_backends = ["cpu", "gpu"] + plugins.keys()

def tf_xla_py_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        main = None,
        enabled_backends = None,
        disabled_backends = None,
        use_xla_device = True,
        enable_mlir_bridge = True,
        **kwargs):
    """Generates py_test targets, one per XLA backend.

    This rule generates py_test() targets named name_backend, for each backend
    in all_backends. The rule also generates a test suite with named `name` that
    tests all backends for the test.

    For example, the following rule generates test cases foo_test_cpu,
    foo_test_gpu, and a test suite name foo_test that tests both.
    tf_xla_py_test(
        name="foo_test",
        srcs="foo_test.py",
        deps=[...],
    )

    Args:
      name: Name of the target.
      srcs: Sources for the target.
      deps: Dependencies of the target.
      tags: Tags to apply to the generated targets.
      data: Data dependencies of the target.
      main: Same as py_test's main attribute.
      enabled_backends: A list of backends that should be tested. Supported
        values include "cpu" and "gpu". If not specified, defaults to None.
      disabled_backends: A list of backends that should not be tested. Supported
        values include "cpu" and "gpu". If not specified, defaults to None.
      use_xla_device: If true then the --test_device argument is set to XLA_CPU
        and XLA_GPU for the CPU and GPU tests.  Otherwise it is set to CPU and
        GPU.
      enable_mlir_bridge: If true, then runs the test with and without mlir
        bridge enabled.
      **kwargs: keyword arguments passed onto the generated py_test() rules.
    """
    if enabled_backends == None:
        enabled_backends = all_backends
    if disabled_backends == None:
        disabled_backends = []
    if type(disabled_backends) != "list":
        fail("disabled_backends must be a list of strings", "disabled_backends")

    backends = [b for b in enabled_backends if b not in disabled_backends]
    test_names = []

    if use_xla_device:
        cpu_xla_device = "XLA_CPU"
        gpu_xla_device = "XLA_GPU"
    else:
        cpu_xla_device = "CPU"
        gpu_xla_device = "GPU"

    for backend in backends:
        test_name = "{}_{}".format(name, backend)
        backend_tags = ["tf_xla_{}".format(backend)]
        backend_args = []
        backend_deps = []
        backend_data = []
        if backend == "cpu":
            backend_args += [
                "--test_device=" + cpu_xla_device,
                "--types=DT_HALF,DT_FLOAT,DT_DOUBLE,DT_UINT8,DT_QUINT8,DT_INT8,DT_QINT8,DT_INT32,DT_QINT32,DT_INT64,DT_BOOL,DT_COMPLEX64,DT_COMPLEX128",
            ]
        elif backend == "gpu":
            backend_args += [
                "--test_device=" + gpu_xla_device,
                "--types=DT_HALF,DT_FLOAT,DT_DOUBLE,DT_UINT8,DT_QUINT8,DT_INT8,DT_QINT8,DT_INT32,DT_QINT32,DT_INT64,DT_BOOL,DT_COMPLEX64,DT_COMPLEX128,DT_BFLOAT16",
            ]
            backend_tags += tf_cuda_tests_tags()
        elif backend in plugins:
            backend_args += [
                "--test_device=" + plugins[backend]["device"],
                "--types=" + plugins[backend]["types"],
            ]
            backend_tags += plugins[backend]["tags"]
            backend_args += plugins[backend]["args"]
            backend_deps += plugins[backend]["deps"]
            backend_data += plugins[backend]["data"]
        else:
            fail("Unknown backend {}".format(backend))

        test_tags = tags + backend_tags

        enable_mlir_bridge_options = [False]
        if enable_mlir_bridge:
            enable_mlir_bridge_options.append(True)

        for mlir_option in enable_mlir_bridge_options:
            extra_dep = []
            extra_tag = []
            updated_name = test_name

            mlir_bridge_dep = "//tensorflow/python:is_mlir_bridge_test_true"
            has_mlir_dep = (mlir_bridge_dep in deps)
            if mlir_option:
                if updated_name.endswith("_test"):
                    updated_name = updated_name[:-5]
                updated_name += "_mlir_bridge_test"
                extra_dep = [] if has_mlir_dep else [mlir_bridge_dep]

                # Mark gpu mlir_bridge tests as ondemand
                #
                # This is for testing book keeping because the bridge does not have any gpu specific
                # logic at this time, so CPU testing is good enough and cheaper.
                extra_tag = ["ondemand"] if backend == "gpu" else []
            elif has_mlir_dep:
                # Some tests run only with mlir_bridge by explicitly adding the MLIR
                # bridge dep so if the dep is already present skip non MLIR
                # version.
                continue

            py_test(
                name = updated_name,
                srcs = srcs,
                srcs_version = "PY3",
                args = backend_args,
                main = "{}.py".format(name) if main == None else main,
                data = data + backend_data,
                deps = deps + backend_deps + extra_dep,
                tags = test_tags + extra_tag,
                exec_properties = tf_exec_properties({"tags": test_tags}),
                **kwargs
            )
            test_names.append(updated_name)
    native.test_suite(name = name, tests = test_names)

def generate_backend_suites(backends = []):
    """Generates per-backend test_suites that run all tests for a backend."""
    if not backends:
        backends = all_backends
    for backend in backends:
        native.test_suite(name = "%s_tests" % backend, tags = ["tf_xla_%s" % backend])
