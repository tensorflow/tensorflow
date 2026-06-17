"""Wrapper around proto libraries used inside the XLA codebase."""

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:dicts.bzl", "dicts")
load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xla:py_strict.bzl", "py_strict_test")
load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_XLA_USERS",
)
load(
    "//xla/tsl:tsl.bzl",
    "tsl_copts",
)
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_exec_properties",
)
load("//xla/tsl/platform/default:build_config.bzl", "strict_cc_test")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_XLA_USERS)

def xla_compile_target_cpu():
    return ""

def xla_py_proto_library(**_kwargs):
    # Note: we don't currently define a proto library target for Python in OSS.
    pass

def xla_py_test_deps():
    return []

def xla_internal_plugin_deps():
    return []

# TODO(ddunleavy): some of these should be removed from here and added to
# specific targets.
# We actually shouldn't need this anymore post vendoring. If we build without
# `framework_shared_object` in the bazelrc all of this should be able to go
# away. The problem is making sure that all these impl deps are `if_static`'d
# appropriately throughout XLA.
_XLA_SHARED_OBJECT_SENSITIVE_DEPS = [
    Label("//xla:autotune_results_proto_cc_impl"),
    Label("//xla:autotuning_proto_cc_impl"),
    Label("//xla:xla_data_proto_cc_impl"),
    Label("//xla:xla_proto_cc_impl"),
    Label("//xla/service:buffer_assignment_proto_cc_impl"),
    Label("//xla/service:hlo_proto_cc_impl"),
    Label("//xla/service:metrics_proto_cc_impl"),
    Label("//xla/service/gpu:backend_configs_cc_impl"),
    Label("//xla/service/gpu/model:hlo_op_profile_proto_cc_impl"),
    Label("//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl"),
    Label("//xla/stream_executor:device_description_proto_cc_impl"),
    Label("//xla/stream_executor:stream_executor_impl"),
    Label("//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl"),
    Label("//xla/backends/cpu/runtime:thunk_proto_cc_impl"),
    "//xla/tsl/framework:allocator_registry_impl",
    "//xla/tsl/framework:allocator",
    "//xla/tsl/platform:env_impl",
    "//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
    "//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
    "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
    "@tsl//tsl/profiler/protobuf:xplane_proto_cc_impl",
    "//xla/tsl/profiler/utils:time_utils_impl",
    "//xla/tsl/protobuf:protos_all_cc_impl",
] + if_rocm_is_configured([
    "//xla/tsl/util:determinism",
])

def xla_cc_binary(deps = [], copts = tsl_copts(), **kwargs):
    cc_binary(deps = deps + _XLA_SHARED_OBJECT_SENSITIVE_DEPS, copts = copts, **kwargs)

def xla_cc_test(
        name,
        deps = [],
        **kwargs):
    """A wrapper around strict_cc_test that adds XLA-specific dependencies.

    Use xla_cc_test or xla_test instead of cc_test in all .../xla/... directories except .../tsl/...,
    where tsl_cc_test should be used.

    Args:
      name: The name of the test.
      deps: The dependencies of the test.
      **kwargs: Other arguments to pass to the test.
    """

    strict_cc_test(
        name = name,
        deps = deps + _XLA_SHARED_OBJECT_SENSITIVE_DEPS,
        exec_properties = tf_exec_properties(kwargs),
        **kwargs
    )

def xla_internal(targets, otherwise = []):
    _ = targets  # buildifier: disable=unused-variable
    return otherwise

def tests_build_defs_bzl_deps():
    return []

def xla_bzl_library(name = "xla_bzl_library"):
    bzl_library(
        name = "xla_bzl",
        srcs = ["xla.default.bzl"],
        deps = [
            "//xla/tsl:tsl_bzl",
            "@local_config_rocm//rocm:build_defs_bzl",
            "//xla/tsl/platform:build_config_root_bzl",
            "//xla/tsl/platform/default:cuda_build_defs_bzl",
            "@bazel_skylib//:bzl_library",
        ],
    )

def _symlink_dynamic_libs_rule_impl(ctx):
    runfiles = ctx.runfiles()
    runfiles_symlinks = {}
    for dep in ctx.attr.deps:
        linker_inputs = dep[CcInfo].linking_context.linker_inputs.to_list()
        for linker_input in linker_inputs:
            if len(linker_input.libraries) == 0:
                continue
            lib = linker_input.libraries[0].dynamic_library
            if not lib:
                continue
            lib_path = paths.join(ctx.attr.lib_dir, lib.basename)
            runfiles_symlinks[lib_path] = lib
    return [
        DefaultInfo(runfiles = ctx.runfiles(
            symlinks = runfiles_symlinks,
        ).merge(runfiles)),
    ]

_symlink_dynamic_libs_rule = rule(
    implementation = _symlink_dynamic_libs_rule_impl,
    attrs = {
        "deps": attr.label_list(allow_empty = True),
        "lib_dir": attr.string(mandatory = True),
    },
    doc = "Symlinks all dynamic libraries for `deps` into a single `lib_dir` directory.",
)

def xla_py_strict_test(name, deps = None, data = None, env = None, need_cuda_libs = False, **kwargs):
    """A wrapper around py_strict_test that adds XLA-specific dependencies.

    Args:
      name: The name of the test.
      deps: The dependencies of the test.
      data: The data dependencies of the test.
      env: The environment variables to set for the test.
      need_cuda_libs: Whether to add CUDA libraries as data dependencies.
      **kwargs: Other arguments to pass to the test.
    """
    deps = deps or []
    data = data or []
    env = env or {}

    if need_cuda_libs:
        library_target = "_{}_libs".format(name)
        lib_dir = paths.join(
            native.package_name(),
            library_target,
        )

        # If the python tests needs to have CUDA libraries as data dependencies, we symlink
        # them into a directory inside the runfiles directory that the test can access and add
        # that directory to the LD_LIBRARY_PATH and CUDA_HOME environment variables.
        _symlink_dynamic_libs_rule(
            name = library_target,
            lib_dir = lib_dir,
            deps = if_cuda_is_configured(
                [
                    "//xla/stream_executor/cuda:all_runtime",
                ],
            ),
            testonly = True,
            visibility = ["//visibility:private"],
        )

        data = data + [library_target]
        env = dicts.add(env, {
            "CUDA_HOME": lib_dir,
            "LD_LIBRARY_PATH": lib_dir,
        })

    py_strict_test(
        name = name,
        deps = deps + xla_py_test_deps(),
        data = data,
        env = env,
        **kwargs
    )
