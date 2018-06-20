"""Build rules for Tensorflow/XLA testing."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_is_configured")
load("//tensorflow/compiler/tests:plugin.bzl", "plugins")

def all_backends():
  b = ["cpu"] + plugins.keys()
  if cuda_is_configured():
    return b + ["gpu"]
  else:
    return b

def tf_xla_py_test(name, srcs=[], deps=[], tags=[], data=[], main=None,
                   disabled_backends=None, **kwargs):
  """Generates py_test targets, one per XLA backend.

  This rule generates py_test() targets named name_backend, for each backend
  in all_backends(). The rule also generates a test suite with named `name` that
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
    disabled_backends: A list of backends that should not be tested. Supported
      values include "cpu" and "gpu". If not specified, defaults to None.
    **kwargs: keyword arguments passed onto the generated py_test() rules.
  """
  if disabled_backends == None:
    disabled_backends = []

  enabled_backends = [b for b in all_backends() if b not in disabled_backends]
  test_names = []
  for backend in enabled_backends:
    test_name = "{}_{}".format(name, backend)
    backend_tags = ["tf_xla_{}".format(backend)]
    backend_args = []
    backend_deps = []
    backend_data = []
    if backend == "cpu":
      backend_args += [
          "--test_device=XLA_CPU",
          "--types=DT_HALF,DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64,DT_BOOL,DT_COMPLEX64"
      ]
    elif backend == "gpu":
      backend_args += [
          "--test_device=XLA_GPU",
          "--types=DT_HALF,DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64,DT_BOOL,DT_COMPLEX64,DT_BFLOAT16"
      ]
      backend_tags += ["requires-gpu-sm35"]
    elif backend in plugins:
      backend_args += ["--test_device=" + plugins[backend]["device"],
                       "--types=" + plugins[backend]["types"]]
      backend_tags += plugins[backend]["tags"]
      backend_args += plugins[backend]["args"]
      backend_deps += plugins[backend]["deps"]
      backend_data += plugins[backend]["data"]
    else:
      fail("Unknown backend {}".format(backend))

    native.py_test(
        name=test_name,
        srcs=srcs,
        srcs_version="PY2AND3",
        args=backend_args,
        main="{}.py".format(name) if main == None else main,
        data=data + backend_data,
        deps=deps + backend_deps,
        tags=tags + backend_tags,
        **kwargs
    )
    test_names.append(test_name)
  native.test_suite(name=name, tests=test_names)

def generate_backend_suites(backends=[]):
  """Generates per-backend test_suites that run all tests for a backend."""
  if not backends:
    backends = all_backends()
  for backend in backends:
    native.test_suite(name="%s_tests" % backend, tags=["tf_xla_%s" % backend])
