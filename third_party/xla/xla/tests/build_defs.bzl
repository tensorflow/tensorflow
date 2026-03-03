"""Build rules for XLA testing. This file is only used for the OSS build and running tests on
github.
"""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "is_rocm_configured",
)
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:plugin.bzl", "plugins")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl:tsl.bzl", "if_google")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//xla/tsl/platform/default:build_config.bzl", "strict_cc_test")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "is_cuda_configured")

visibility(DEFAULT_LOAD_VISIBILITY)

# Possible backend values for the GPU family.
NVIDIA_GPU_BACKENDS = [
    "nvgpu_any",
    "p100",
    "v100",
    "a100",
    "h100",
    "b200",
    "gb200",
]

# The generic "gpu" backend includes the actual backends in this list.
NVIDIA_GPU_DEFAULT_BACKENDS = [
    "nvgpu_any",
    "a100",
    "h100",
    "b200",
    "gb200",
]

AMD_GPU_DEFAULT_BACKENDS = ["amdgpu_any"]

INTEL_GPU_DEFAULT_BACKENDS = ["intelgpu_any"]

_DEFAULT_BACKENDS = ["cpu"] + NVIDIA_GPU_DEFAULT_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS

GPU_BACKENDS = NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS

GPU_DEFAULT_BACKENDS = NVIDIA_GPU_DEFAULT_BACKENDS

DEFAULT_DISABLED_BACKENDS = []

_ALL_BACKENDS = ["cpu", "interpreter"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS + list(plugins.keys())

# buildifier: disable=function-docstring
def prepare_nvidia_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    # Expand "gpu" backend name into device specific backend names unless it's tagged rocm-only or oneapi-only
    new_backends = [name for name in backends if name != "gpu"]
    if len(new_backends) < len(backends) and "rocm-only" not in common_tags and "oneapi-only" not in common_tags:
        new_backends.extend(NVIDIA_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(NVIDIA_GPU_BACKENDS)

    new_backend_tags = {key: value for key, value in backend_tags.items() if key != "gpu"}
    gpu_backend_tags = backend_tags.get("gpu", tf_gpu_tests_tags())
    for key in NVIDIA_GPU_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    new_backend_args = {key: value for key, value in backend_args.items() if key != "gpu"}
    if "gpu" in backend_args:
        for key in NVIDIA_GPU_BACKENDS:
            new_backend_args.setdefault(key, backend_args["gpu"])

    # Disable backends that don't meet the device requirements.
    sm_requirements = {
        "nvgpu_any": (0, 0),
        "p100": (6, 0),
        "v100": (7, 0),
        "a100": (8, 0),
        "h100": (9, 0),
        "b200": (10, 0),
        "gb200": (10, 0),
    }
    for gpu_backend in NVIDIA_GPU_BACKENDS:
        all_tags = new_backend_tags[gpu_backend]
        requires_gpu = [t for t in all_tags if t.startswith("requires-gpu-")]

        # full tag is meaningless in OSS, when run at google, this routes tests to full GPUs
        # instead of mig partition(s).
        requires_full_gpu = if_google("full" in new_backend_tags[gpu_backend], False)
        requires_sm, only = None, False
        num_gpus = None
        for tag in requires_gpu:
            if ":" in tag:  # Multi-GPU tests are suffixed with colon and number of GPUs.
                tag, suffix = tag.split(":")  # Remove the suffix from the tag for further parsing.
                parsed_num_gpus = int(suffix)
                if num_gpus and num_gpus != parsed_num_gpus:
                    fail("Inconsistent number of GPUs: %d vs %d" % (num_gpus, parsed_num_gpus))
                num_gpus = parsed_num_gpus
            if tag.startswith("requires-gpu-sm"):
                version = tag.split("-")[2][2:]
                sm = (int(version[:-1]), int(version[-1]))
                if not requires_sm or sm < requires_sm:
                    requires_sm = sm
                if tag.endswith("-only"):
                    only = True
        if only:
            disable = requires_sm != sm_requirements[gpu_backend]
        else:
            disable = requires_sm and requires_sm > sm_requirements[gpu_backend]

        if disable:
            new_disabled_backends.append(gpu_backend)
        else:
            sm_major, sm_minor = sm_requirements[gpu_backend]
            full = "-full" if requires_full_gpu else ""
            sm_tag = "requires-gpu-nvidia" if sm_major == 0 else "requires-gpu-sm%s%s%s-only" % (sm_major, sm_minor, full)
            if num_gpus:
                sm_tag += ":%d" % num_gpus
            new_backend_tags[gpu_backend] = [t for t in all_tags if t not in requires_gpu]
            new_backend_tags[gpu_backend].append(sm_tag)
            new_backend_tags[gpu_backend].append("cuda-only")

    return new_backends, new_disabled_backends, new_backend_tags, new_backend_args

# buildifier: disable=function-docstring
def prepare_amd_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    new_backends = [name for name in backends if name != "gpu"]

    # Expand "gpu" backend name into device specific backend names unless it's tagged cuda-only or oneapi-only
    if len(new_backends) < len(backends) and "cuda-only" not in common_tags and "oneapi-only" not in common_tags:
        new_backends.extend(AMD_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(AMD_GPU_DEFAULT_BACKENDS)

    new_backend_tags = {
        key: value
        for key, value in backend_tags.items()
        if key not in ["gpu"] + NVIDIA_GPU_BACKENDS
    }

    gpu_backend_tags = backend_tags.get("gpu", [])
    nvidia_tags = []
    for key in gpu_backend_tags:
        if key.startswith("requires-"):
            nvidia_tags.append(key)

    for key in nvidia_tags:
        gpu_backend_tags.remove(key)

    for key in AMD_GPU_DEFAULT_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    for backend in AMD_GPU_DEFAULT_BACKENDS:
        new_backend_tags[backend].append("requires-gpu-amd")
        new_backend_tags[backend].append("rocm-only")

    return new_backends, new_disabled_backends, new_backend_tags, backend_args

# buildifier: disable=function-docstring
def prepare_intel_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    new_backends = [name for name in backends if name != "gpu"]

    # Expand "gpu" backend name into device specific backend names unless it's tagged cuda-only or rocm-only
    if len(new_backends) < len(backends) and "cuda-only" not in common_tags and "rocm-only" not in common_tags:
        new_backends.extend(INTEL_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(INTEL_GPU_DEFAULT_BACKENDS)

    new_backend_tags = {
        key: value
        for key, value in backend_tags.items()
        if key not in ["gpu"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS
    }

    gpu_backend_tags = backend_tags.get("gpu", [])
    nvidia_tags = []
    for key in gpu_backend_tags:
        if key.startswith("requires-"):
            nvidia_tags.append(key)

    for key in nvidia_tags:
        gpu_backend_tags.remove(key)

    for key in INTEL_GPU_DEFAULT_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    for backend in INTEL_GPU_DEFAULT_BACKENDS:
        new_backend_tags[backend].append("requires-gpu-intel")
        new_backend_tags[backend].append("oneapi-only")

    return new_backends, new_disabled_backends, new_backend_tags, backend_args

# buildifier: disable=function-docstring
def prepare_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    nvidia_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + NVIDIA_GPU_BACKENDS
    ]
    amd_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + AMD_GPU_DEFAULT_BACKENDS
    ]
    intel_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + INTEL_GPU_DEFAULT_BACKENDS
    ]
    other_backends = [
        backend
        for backend in backends
        if backend not in ["gpu"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS
    ]

    nvidia_backends, nvidia_disabled_backends, nvidia_backend_tags, nvidia_backend_args = \
        prepare_nvidia_gpu_backend_data(nvidia_backends, disabled_backends, backend_tags, backend_args, common_tags)
    amd_backends, amd_disabled_backends, amd_backend_tags, amd_backend_args = \
        prepare_amd_gpu_backend_data(amd_backends, disabled_backends, backend_tags, {}, common_tags)
    intel_backends, intel_disabled_backends, intel_backend_tags, intel_backend_args = \
        prepare_intel_gpu_backend_data(intel_backends, disabled_backends, backend_tags, {}, common_tags)

    new_backends = [
        backend
        for backend in nvidia_backends + amd_backends + intel_backends + other_backends
    ]

    disabled_backends = nvidia_disabled_backends + amd_disabled_backends + intel_disabled_backends

    backend_tags = nvidia_backend_tags | amd_backend_tags | intel_backend_tags

    backend_args = nvidia_backend_args | amd_backend_args | intel_backend_args

    return new_backends, disabled_backends, backend_tags, backend_args

def xla_test(
        name,
        srcs,
        deps,
        backends = [],
        disabled_backends = DEFAULT_DISABLED_BACKENDS,
        real_hardware_only = False,  # @unused, all backends are real hardware.
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
      real_hardware_only: No-op.
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
        this_backend_kwargs = dict(kwargs) | backend_kwargs.get(backend, {})
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

        xla_cc_test(
            name = test_name,
            srcs = srcs,
            tags = this_backend_tags,
            copts = copts + this_backend_copts,
            args = args + this_backend_args,
            deps = deps + backend_deps,
            data = data + this_backend_data,
            env = env | {
                "XLA_TEST_DEVICE": device,
                "XLA_TEST_DEVICE_TYPE": device_type_for_env,
                "XLA_TEST_MODIFIERS": ",".join(modifiers),
            },
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
