# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Build rules for XLA testing. This file is only used for the OSS build and running tests on
github.
"""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "is_rocm_configured",
)
load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:xla.default.bzl", "xla_cc_test", "xla_py_test_deps")
load(
    "//xla/tests:backend_defs.bzl",
    _ALL_BACKENDS = "ALL_BACKENDS",
    _ALL_HARDWARE_BACKENDS = "ALL_HARDWARE_BACKENDS",
    _AMD_GPU_DEFAULT_BACKENDS = "AMD_GPU_DEFAULT_BACKENDS",
    _DEFAULT_BACKENDS = "DEFAULT_BACKENDS",
    _DEFAULT_DISABLED_BACKENDS = "DEFAULT_DISABLED_BACKENDS",
    _GPU_BACKENDS = "GPU_BACKENDS",
    _GPU_DEFAULT_BACKENDS = "GPU_DEFAULT_BACKENDS",
    _INTEL_GPU_DEFAULT_BACKENDS = "INTEL_GPU_DEFAULT_BACKENDS",
    _NVIDIA_GPU_BACKENDS = "NVIDIA_GPU_BACKENDS",
    _NVIDIA_GPU_DEFAULT_BACKENDS = "NVIDIA_GPU_DEFAULT_BACKENDS",
    _prepare_gpu_backend_data = "prepare_gpu_backend_data",
)
load("//xla/tests:plugin.bzl", "plugins")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//xla/tsl/platform/default:build_config.bzl", "strict_cc_test")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "is_cuda_configured")

visibility(DEFAULT_LOAD_VISIBILITY)

ALL_HARDWARE_BACKENDS = _ALL_HARDWARE_BACKENDS
AMD_GPU_DEFAULT_BACKENDS = _AMD_GPU_DEFAULT_BACKENDS
DEFAULT_DISABLED_BACKENDS = _DEFAULT_DISABLED_BACKENDS
GPU_BACKENDS = _GPU_BACKENDS
GPU_DEFAULT_BACKENDS = _GPU_DEFAULT_BACKENDS
INTEL_GPU_DEFAULT_BACKENDS = _INTEL_GPU_DEFAULT_BACKENDS
NVIDIA_GPU_BACKENDS = _NVIDIA_GPU_BACKENDS
NVIDIA_GPU_DEFAULT_BACKENDS = _NVIDIA_GPU_DEFAULT_BACKENDS
prepare_gpu_backend_data = _prepare_gpu_backend_data

def xla_test(
        name,
        srcs,
        deps,
        backends = [],
        disabled_backends = DEFAULT_DISABLED_BACKENDS,
        args = [],
        tags = [],
        copts = [],
        data = [],
        env = {},
        backend_tags = {},
        backend_args = {},
        backend_kwargs = {},
        linkstatic = None,
        fail_if_no_test_linked = True,
        fail_if_no_test_selected = True,
        use_legacy_runtime = False,
        test_cpu_fast_compile = False,
        **kwargs):
    """Generates strict_cc_test targets for the given XLA backends.

    This rule is similar to platforms/.../build_defs.bzl but only meant for running the tests on
    github.

    This rule generates a cc_test target for one or more XLA backends. The arguments
    are identical to cc_test with two additions: 'backends' and 'backend_args'.
    'backends' specifies the backends to generate tests for ("cpu", "gpu"), and
    'backend_args'/'backend_tags' specifies backend-specific args parameters to use
    when generating the cc_test.

    The name of the cc_tests are the provided name argument with the backend name
    appended. For example, if name parameter is "foo_test", then the cpu
    test target will be "foo_test_cpu".

    The build rule also defines a test suite ${name} which includes the tests for
    each of the supported backends.

    Each generated cc_test target has a tag indicating which backend the test is
    for. This tag is of the form "xla_${BACKEND}" (eg, "xla_cpu"). These
    tags can be used to gather tests for a particular backend into a test_suite.

    Use xla_test instead of cc_test or xla_cc_test in all tests that need to run
    on specific XLA backends. Do not use xla_test in .../tsl/... directories,
    where tsl_cc_test should be used instead.

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
      backends: A list of backends to generate tests for. Supported values: "cpu",
        "gpu". If this list is empty, the test will be generated for all supported
        backends.
      disabled_backends: A list of backends to NOT generate tests for.
      args: Test arguments for the target.
      tags: Tags for the target.
      copts: Additional copts to pass to the build.
      data: Additional data to pass to the build.
      env: Env vars to set for the test.
      backend_tags: A dict mapping backend name to list of additional tags to
        use for that target.
      backend_args: A dict mapping backend name to list of additional args to
        use for that target.
      backend_kwargs: A dict mapping backend name to list of additional keyword
        arguments to pass to strict_cc_test. Only use for kwargs that don't have a
        dedicated argument, like setting per-backend flaky or timeout attributes.
      linkstatic: Whether to link the test statically. Can be set to None to use
        the default value decided by strict_cc_test.
      fail_if_no_test_linked: Whether to fail if no test case is linked into the test.
      fail_if_no_test_selected: Whether to fail if no test case is executed.
      use_legacy_runtime: If true, adds the required dependencies for writing tests
        using the legacy runtime.
      test_cpu_fast_compile: If true, generate a specialized test target for CPU
        with FAST_COMPILE preset enabled.
      **kwargs: Additional keyword arguments to pass to strict_cc_test.
    """

    # aot_compile_test is not supported in OSS.
    kwargs.pop("aot_compile_test", None)

    # TODO: b/382779188 - Remove this once all legacy tests have had this kwarg added.
    kwargs.pop("use_legacy_runtime", None)

    test_names = []
    if not backends:
        backends = _DEFAULT_BACKENDS

    # Expand "gpu" backend name to specific GPU backends and update tags.
    backends, disabled_backends, backend_tags, backend_args = \
        prepare_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, tags)

    backends = [
        backend
        for backend in backends
        if backend not in disabled_backends
    ]

    for backend in backends:
        test_name = "%s_%s" % (name, backend)
        this_backend_tags = ["xla_%s" % backend] + tags + backend_tags.get(backend, [])
        this_backend_copts = []
        this_backend_args = backend_args.get(backend, [])
        this_backend_kwargs = dict(kwargs)
        for k, v in backend_kwargs.get(backend, {}).items():
            this_backend_kwargs[k] = v
        this_backend_data = []
        backend_deps = []
        if backend == "cpu":
            device_type_for_env = "cpu"
            backend_deps.append("//xla/service:cpu_plugin")

            if not use_legacy_runtime:
                backend_deps.append("//xla/tests:pjrt_cpu_client_registry")
        elif backend in NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS:
            device_type_for_env = "gpu"
            backend_deps.append(
                "//xla/service:gpu_plugin",
            )
            if backend in NVIDIA_GPU_BACKENDS:
                this_backend_tags += tf_gpu_tests_tags()
                backend_deps += [
                    "//xla/stream_executor/cuda:all_runtime",
                    "//xla/stream_executor/cuda:gpu_test_kernels_cuda",
                    "//xla/stream_executor/cuda:stream_executor_cuda",
                ]
            if backend in AMD_GPU_DEFAULT_BACKENDS:
                this_backend_tags.append("gpu")
                if "multi_gpu" in this_backend_tags:
                    this_backend_tags.append("exclusive-if-local")
                backend_deps += [
                    "//xla/stream_executor/rocm:all_runtime",
                    "//xla/stream_executor/rocm:gpu_test_kernels_rocm",
                    "//xla/stream_executor/rocm:stream_executor_rocm",
                ]
            if backend in INTEL_GPU_DEFAULT_BACKENDS:
                this_backend_tags.append("gpu")
                backend_deps += [
                    "//xla/stream_executor/sycl:all_runtime",
                    "//xla/stream_executor/sycl:stream_executor_sycl",
                ]

            if not use_legacy_runtime:
                backend_deps.append("//xla/tests:pjrt_gpu_client_registry")
        elif backend == "interpreter":
            device_type_for_env = "interpreter"
            backend_deps.append(
                "//xla/service:interpreter_plugin",
            )

            if not use_legacy_runtime:
                backend_deps.append("//xla/tests:pjrt_interpreter_client_registry")
        elif backend in plugins:
            device_type_for_env = "plugin"
            backend_deps += plugins[backend]["deps"]
            this_backend_copts += plugins[backend]["copts"]
            this_backend_tags += plugins[backend]["tags"]
            this_backend_args += plugins[backend]["args"]
            this_backend_data += plugins[backend]["data"]
        else:
            fail("Unknown backend %s" % backend)

        # Ensure that the tags are consistent with the runtime used.
        if "pjrt_migration_candidate" in this_backend_tags and not use_legacy_runtime:
            fail("xla_tests that do not use the legacy runtime configuration should not be tagged `pjrt_migration_candidate`.")
        if "test_migrated_to_hlo_runner_pjrt" in this_backend_tags:
            fail("The `test_migrated_to_hlo_runner_pjrt` tag is deprecated and should no longer be used.")

        modifiers = backend.split("_")
        device = modifiers.pop(0)

        this_backend_env = dict(env)
        for k, v in {
            "XLA_TEST_DEVICE": device,
            "XLA_TEST_DEVICE_TYPE": device_type_for_env,
            "XLA_TEST_MODIFIERS": ",".join(modifiers),
        }.items():
            this_backend_env[k] = v

        xla_cc_test(
            name = test_name,
            srcs = srcs,
            tags = this_backend_tags,
            copts = copts + this_backend_copts,
            args = args + this_backend_args,
            deps = deps + backend_deps,
            data = data + this_backend_data,
            env = this_backend_env,
            linkstatic = linkstatic,
            fail_if_no_test_linked = fail_if_no_test_linked,
            fail_if_no_test_selected = fail_if_no_test_selected,
            **this_backend_kwargs
        )

        if backend == "cpu" and test_cpu_fast_compile:
            fast_compile_test_name = test_name + "_fast_compile"
            fast_compile_env = dict(env)
            for k, v in {
                "XLA_TEST_DEVICE": device,
                "XLA_TEST_DEVICE_TYPE": device_type_for_env,
                "XLA_TEST_MODIFIERS": ",".join(modifiers),
            }.items():
                fast_compile_env[k] = v
            if "XLA_FLAGS" in fast_compile_env:
                fast_compile_env["XLA_FLAGS"] += " --xla_cpu_opt_preset=FAST_COMPILE"
            else:
                fast_compile_env["XLA_FLAGS"] = "--xla_cpu_opt_preset=FAST_COMPILE"

            xla_cc_test(
                name = fast_compile_test_name,
                srcs = srcs,
                tags = this_backend_tags,
                copts = copts + this_backend_copts,
                args = args + this_backend_args,
                deps = deps + backend_deps,
                data = data + this_backend_data,
                env = fast_compile_env,
                linkstatic = linkstatic,
                fail_if_no_test_linked = fail_if_no_test_linked,
                fail_if_no_test_selected = fail_if_no_test_selected,
                **this_backend_kwargs
            )

        if ((backend in NVIDIA_GPU_BACKENDS and is_cuda_configured()) or
            (backend in AMD_GPU_DEFAULT_BACKENDS and is_rocm_configured())):
            test_names.append(test_name)

    # Notably, a test_suite with `tests = []` is not empty:
    # https://bazel.build/reference/be/general#test_suite_args and the default
    # `tests = []` behavior doesn't respect `--build_tag_filters` due to
    # b/317293391. For this reason, if we would create an empty `test_suite`,
    # instead create a `cc_test` with no srcs that links against `main` to have
    # more predictable behavior that avoids bugs.
    #
    # Due to b/317293391, we also mark the test suite `manual`, so that wild card builds
    # like in the XLA CI won't try to build the test suite target. Instead the wild card
    # build will build the individual test targets and therefore respect the tags on each
    # individual test target.
    # Example: Assume we have an `xla_test(name=my_test)` in `//xla/service/gpu` with backends `cpu`
    # and `gpu`. This generates two test targets `//xla/service/gpu:my_test_{cpu|gpu}`. The latter
    # has a tag `gpu`.
    #
    # - `bazel test --test_tag_filters=-gpu //xla/service/gpu/...` will only run the cpu test.
    # - `bazel test //xla/service/gpu/...` will run both tests.
    # - `bazel test //xla/service/gpu:my_test` will run both tests.
    # Caveat:
    # - `bazel test --test_tag_filters=-gpu //xla/service/gpu:my_test` will run both tests and
    #   not respect the tag filter - but it's way better than the previous behavoir.
    if test_names:
        native.test_suite(name = name, tags = tags + ["manual"], tests = test_names)
    else:
        strict_cc_test(
            name = name,
            deps = ["@com_google_googletest//:gtest_main"],
            linkstatic = linkstatic,
            # This test is deliberately empty. Its only purpose is to avoid
            # creating an empty test suite, which would be a problem for
            # --build_tag_filters (see above). Therefore we don't want to fail
            # if no test case is linked in or runs.
            fail_if_no_test_linked = False,
            fail_if_no_test_selected = False,
        )

def generate_backend_suites(backends = []):  # buildifier: disable=unnamed-macro
    """Generates test_suites containing all tests for each backend.

    Generates test_suites of the form "${backend}_tests" containing all tests
    matching that backend for all tests in the package the macro is called in.

    Args:
      backends: The list of backends to generate test_suites for.
    """

    if not backends:
        backends = _ALL_BACKENDS
    for backend in backends:
        native.test_suite(
            name = "%s_tests" % backend,
            tags = ["xla_%s" % backend, "-broken", "manual"],
        )

def xla_py_test(name, deps = None, data = None, env = None, **kwargs):
    """A wrapper around py_strict_test that adds XLA-specific dependencies.

    Args:
      name: The name of the test.
      deps: The dependencies of the test.
      data: The data dependencies of the test.
      env: The environment variables to set for the test.
      **kwargs: Other arguments to pass to the test.
    """
    deps = deps or []
    data = data or []
    env = env or {}

    py_strict_test(
        name = name,
        deps = deps + xla_py_test_deps(),
        data = data,
        env = env,
        **kwargs
    )
