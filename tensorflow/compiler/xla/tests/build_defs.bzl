"""Build rules for XLA testing."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_is_configured")

def all_backends():
  if cuda_is_configured():
    return ["cpu", "cpu_parallel", "gpu"]
  else:
    return ["cpu", "cpu_parallel"]

def xla_test(name,
             srcs,
             deps,
             backends=[],
             args=[],
             tags=[],
             copts=[],
             backend_tags={},
             backend_args={},
             **kwargs):
  """Generates cc_test targets for the given XLA backends.

  This rule generates a cc_test target for one or more XLA backends and also
  a platform-agnostic cc_library rule. The arguments are identical to cc_test
  with two additions: 'backends' and 'backend_args'. 'backends' specifies the
  backends to generate tests for ("cpu", "cpu_parallel", "gpu"), and
  'backend_args'/'backend_tags' specifies backend-specific args parameters to
  use when generating the cc_test.

  The name of the cc_tests are the provided name argument with the backend name
  appended, and the cc_library target name is the provided name argument with
  "_lib" appended. For example, if name parameter is "foo_test", then the cpu
  test target will be "foo_test_cpu" and the cc_library target is "foo_lib".

  The cc_library target can be used to link with other plugins outside of
  xla_test.

  The build rule also defines a test suite ${name} which includes the tests for
  each of the supported backends.

  Each generated cc_test target has a tag indicating which backend the test is
  for. This tag is of the form "xla_${BACKEND}" (eg, "xla_cpu"). These
  tags can be used to gather tests for a particular backend into a test_suite.

  Examples:

    # Generates the targets: foo_test_cpu and foo_test_gpu.
    xla_test(
        name = "foo_test",
        srcs = ["foo_test.cc"],
        backends = ["cpu", "gpu"],
        deps = [...],
    )

    # Generates the targets: bar_test_cpu and bar_test_gpu. bar_test_cpu
    # includes the additional arg "--special_cpu_flag".
    xla_test(
        name = "bar_test",
        srcs = ["bar_test.cc"],
        backends = ["cpu", "gpu"],
        backend_args = {"cpu": ["--special_cpu_flag"]}
        deps = [...],
    )

  The build rule defines the preprocessor macro XLA_TEST_BACKEND_${BACKEND}
  to the value 1 where ${BACKEND} is the uppercase name of the backend.

  Args:
    name: Name of the target.
    srcs: Sources for the target.
    deps: Dependencies of the target.
    backends: A list of backends to generate tests for. Supported
      values: "cpu", "cpu_parallel", "gpu". If this list is empty, the test will
      be generated for all supported backends.
    args: Test arguments for the target.
    tags: Tags for the target.
    backend_args: A dict mapping backend name to list of additional args to
      use for that target.
    backend_tags: A dict mapping backend name to list of additional tags to
      use for that target.
  """
  test_names = []
  if not backends:
    backends = all_backends()

  native.cc_library(
      name="%s_lib" % name,
      srcs=srcs,
      copts=copts,
      testonly=True,
      deps=deps + ["//tensorflow/compiler/xla/tests:test_macros_header"],
  )

  for backend in backends:
    test_name = "%s_%s" % (name, backend)
    this_backend_tags = ["xla_%s" % backend]
    this_backend_copts = []
    this_backend_args = backend_args.get(backend, [])
    if backend == "cpu":
      backend_deps = ["//tensorflow/compiler/xla/service:cpu_plugin"]
      backend_deps += ["//tensorflow/compiler/xla/tests:test_macros_cpu"]
    elif backend == "cpu_parallel":
      backend_deps = ["//tensorflow/compiler/xla/service:cpu_plugin"]
      backend_deps += ["//tensorflow/compiler/xla/tests:test_macros_cpu"]
      this_backend_args += ["--xla_backend_extra_options=\"xla_cpu_parallel\""]
    elif backend == "gpu":
      backend_deps = ["//tensorflow/compiler/xla/service:gpu_plugin"]
      backend_deps += ["//tensorflow/compiler/xla/tests:test_macros_gpu"]
      this_backend_tags += ["requires-gpu-sm35"]
    else:
      fail("Unknown backend %s" % backend)

    native.cc_test(
        name=test_name,
        srcs=srcs,
        tags=tags + backend_tags.get(backend, []) + this_backend_tags,
        copts=copts + ["-DXLA_TEST_BACKEND_%s=1" % backend.upper()] +
        this_backend_copts,
        args=args + this_backend_args,
        deps=deps + backend_deps,
        **kwargs)

    test_names.append(test_name)

  native.test_suite(name=name, tests=test_names)


def generate_backend_suites(backends=[]):
  if not backends:
    backends = all_backends()
  for backend in backends:
    native.test_suite(name="%s_tests" % backend,
                      tags = ["xla_%s" % backend])


def generate_backend_test_macros(backends=[]):
  if not backends:
    backends = all_backends()
  for backend in backends:
    native.cc_library(
        name="test_macros_%s" % backend,
        testonly = True,
        hdrs = ["test_macros.h"],
        copts = ["-DXLA_PLATFORM=\\\"%s\\\"" % backend.upper()],
        deps = [
            "//tensorflow/compiler/xla:types",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
        ])
