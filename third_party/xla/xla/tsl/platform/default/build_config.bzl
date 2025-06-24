"""Platform-specific build configurations."""

# This file is used in OSS only. It is not transformed by copybara. Therefore all paths in this
# file are OSS paths.

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "clean_dep",
    "if_tsl_link_protobuf",
)
load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_static")
load("@rules_python//python:py_library.bzl", "py_library")

# IMPORTANT: Do not remove this load statement. We rely on that //xla/tsl doesn't exist in g3
# to prevent g3 .bzl files from loading this file.
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def well_known_proto_libs():
    """Set of standard protobuf protos, like Any and Timestamp.

    This list should be provided by protobuf.bzl, but it's not.
    """
    return [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:api_proto",
        "@com_google_protobuf//:compiler_plugin_proto",
        "@com_google_protobuf//:descriptor_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:empty_proto",
        "@com_google_protobuf//:field_mask_proto",
        "@com_google_protobuf//:source_context_proto",
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:type_proto",
        "@com_google_protobuf//:wrappers_proto",
    ]

def tf_deps(deps, suffix):
    """Appends a suffix to a list of deps.

    Args:
      deps: the list of deps which will be suffixed
      suffix: the suffix to add

    Returns:
      The list of deps with the suffix applied.
    """
    tf_deps = []

    # If the package name is in shorthand form (ie: does not contain a ':'),
    # expand it to the full name.
    for dep in deps:
        tf_dep = dep

        if not ":" in dep:
            dep_pieces = dep.split("/")
            tf_dep += ":" + dep_pieces[len(dep_pieces) - 1]

        tf_deps.append(tf_dep + suffix)

    return tf_deps

# Modified from @cython//:Tools/rules.bzl
def pyx_library(
        name,
        cc_deps = [],
        py_deps = [],
        srcs = [],
        testonly = None,
        srcs_version = "PY3",
        copts = [],
        **kwargs):
    """Compiles a group of .pyx / .pxd / .py files.

    First runs Cython to create .cpp files for each input .pyx or .py + .pxd
    pair. Then builds a shared object for each, passing "cc_deps" to each cc_binary
    rule (includes Python headers by default). Finally, creates a py_library rule
    with the shared objects and any pure Python "srcs", with py_deps as its
    dependencies; the shared objects can be imported like normal Python files.

    Args:
      name: Name for the rule.
      cc_deps: C/C++ dependencies of the Cython (e.g. Numpy headers).
      py_deps: Pure Python dependencies of the final library.
      srcs: .py, .pyx, or .pxd files to either compile or pass through.
      testonly: If True, the target can only be used with tests.
      srcs_version: Version of python source files.
      copts: List of copts to pass to cc rules.
      **kwargs: Extra keyword arguments passed to the py_library.
    """

    # First filter out files that should be run compiled vs. passed through.
    py_srcs = []
    pyx_srcs = []
    pxd_srcs = []
    for src in srcs:
        if src.endswith(".pyx") or (src.endswith(".py") and
                                    src[:-3] + ".pxd" in srcs):
            pyx_srcs.append(src)
        elif src.endswith(".py"):
            py_srcs.append(src)
        else:
            pxd_srcs.append(src)
        if src.endswith("__init__.py"):
            pxd_srcs.append(src)

    # Invoke cython to produce the shared object libraries.
    for filename in pyx_srcs:
        native.genrule(
            name = filename + "_cython_translation",
            srcs = [filename],
            outs = [filename.split(".")[0] + ".cpp"],
            # Optionally use PYTHON_BIN_PATH on Linux platforms so that python 3
            # works. Windows has issues with cython_binary so skip PYTHON_BIN_PATH.
            cmd = "PYTHONHASHSEED=0 $(location @cython//:cython_binary) --cplus $(SRCS) --output-file $(OUTS)",
            testonly = testonly,
            tools = ["@cython//:cython_binary"] + pxd_srcs,
        )

    shared_objects = []
    for src in pyx_srcs:
        stem = src.split(".")[0]
        shared_object_name = stem + ".so"
        native.cc_binary(
            name = shared_object_name,
            srcs = [stem + ".cpp"],
            deps = cc_deps + ["@local_xla//third_party/python_runtime:headers"],
            linkshared = 1,
            testonly = testonly,
            copts = copts,
        )
        shared_objects.append(shared_object_name)

    # Now create a py_library with these shared objects as data.
    py_library(
        name = name,
        srcs = py_srcs,
        deps = py_deps,
        srcs_version = srcs_version,
        data = shared_objects,
        testonly = testonly,
        **kwargs
    )

def tf_jspb_proto_library(**_kwargs):
    pass

def tf_proto_library(
        name,
        srcs = [],
        deps = [],
        has_services = None,
        protodeps = [],
        testonly = 0,
        create_grpc_library = False,
        exports = [],
        tags = [],
        visibility = None,
        compatible_with = None,
        cc_grpc_version = None,  # @unused
        use_grpc_namespace = False,  # @unused
        cc_libs = [],  # @unused
        local_defines = None,  # @unused
        make_default_target_header_only = False,  # @unused
        j2objc_api_version = 1,  # @unused
        js_codegen = "jspb",  # @unused
        create_service = False,  # @unused
        create_java_proto = False,  # @unused
        create_kotlin_proto = False,  # @unused
        create_go_proto = False):  # @unused
    """A macro generating protobuf and/or gRPC stubs for C++ and Python.

    It is a backward-compatible (with old TF-custom protobuf and gGRPC rules) macro which wraps a
    bunch of standard rules to create protobuf and gRPC stubs, while preserving backward-compatible
    naming scheme for the targets. The generated public targets are as follows:
       - proto_library:    "{name}"
       - cc_proto_library: "{name}_cc"
       - alias:            "{name}_cc_impl", where actual = "{name}_cc"
       - py_proto_library: "{name}_py_proto"
       - py_grcp_library:  "{name}_py_grpc_proto"
       - py_library:       "{name}_py", where deps = ["{name}_py_proto", "{name}_py_grpc_proto"]
       - cc_grpc_library:  "{name}_cc_grpc_proto", where name = name[:-6] if name.endsWith("proto")

    The header-only gargets are not generated anymore due to being obsolete.

    Args:
      name: The name of the proto_library target.
      srcs: The .proto files to compile.
      deps: The proto_library targets that the srcs depend on.
      has_services: Whether to create Python gRPC stubs.
      protodeps: Additional deps, deprecated, use deps directly instead.
      testonly: Whether the target is testonly.
      create_grpc_library: Whether to create C++ gRPC stubs.
      exports: List of proto_library targets that can be referenced via "import public" in the proto
          source.
      tags: The tags to pass to the proto_library target.
      visibility: The visibility argument to pass to all of the generated targets.
      compatible_with: The compatible_with argument to pass to all of the generated targets.
      cc_grpc_version: Obsolete.
      use_grpc_namespace: Obsolete.
      cc_libs: Obsolete.
      local_defines: Obsolete.
      make_default_target_header_only: Obsolete.
      j2objc_api_version: Obsolete.
      js_codegen: Obsolete.
      create_service: Obsolete.
      create_java_proto: Obsolete.
      create_kotlin_proto: Obsolete.
      create_go_proto: Obsolete.
    """

    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs + deps + protodeps,
        testonly = testonly,
        visibility = visibility,
    )

    if name.endswith("_proto"):
        name_sans_proto = name[:-6]
    else:
        name_sans_proto = name
    native.proto_library(
        name = name,
        srcs = srcs,
        deps = deps + protodeps + well_known_proto_libs(),
        exports = exports,
        compatible_with = compatible_with,
        visibility = visibility,
        testonly = testonly,
        tags = tags,
    )

    cc_proto_name = name + "_cc"
    cc_proto_library(
        name = cc_proto_name,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
        deps = [":{}".format(name)],
    )

    native.alias(
        name = name + "_cc_impl",
        testonly = testonly,
        actual = name + "_cc",
        compatible_with = compatible_with,
        visibility = visibility,
    )

    # TODO(b/356020232): remove completely after migration is done
    # This is strictly speaking incorrect, we need to remove all the
    # references to the _cc_headers_only targets instead
    native.alias(
        name = name + "_cc_headers_only",
        testonly = testonly,
        actual = name + "_cc",
        compatible_with = compatible_with,
        visibility = visibility,
    )

    py_deps = []
    py_proto_name = name + "_py_proto"
    py_deps.append(":{}".format(py_proto_name))
    py_proto_library(
        name = py_proto_name,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
        deps = [":{}".format(name)],
    )

    if has_services:
        py_grpc_name = name + "_py_grpc_proto"
        py_deps.append(":{}".format(py_grpc_name))
        py_grpc_library(
            name = name + "_py_grpc_proto",
            srcs = [":{}".format(name)],
            deps = [":{}".format(py_proto_name)],
            testonly = testonly,
            compatible_with = compatible_with,
            visibility = visibility,
        )

    py_library(
        name = name + "_py",
        srcs = py_deps,
        deps = py_deps,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
    )

    if create_grpc_library:
        cc_grpc_library(
            name = name_sans_proto + "_cc_grpc_proto",
            testonly = testonly,
            srcs = [":{}".format(name)],
            generate_mocks = True,
            visibility = visibility,
            compatible_with = compatible_with,
            deps = [":{}".format(cc_proto_name)],
            plugin_flags = ["services_namespace=grpc"],
            grpc_only = True,
        )

def tf_additional_lib_hdrs():
    return [
        clean_dep("//xla/tsl/platform/default:casts.h"),
        clean_dep("//xla/tsl/platform/default:context.h"),
        clean_dep("//xla/tsl/platform/default:criticality.h"),
        clean_dep("//xla/tsl/platform/default:integral_types.h"),
        clean_dep("//xla/tsl/platform/default:logging.h"),
        clean_dep("//xla/tsl/platform/default:stacktrace.h"),
        clean_dep("//xla/tsl/platform/default:status.h"),
        clean_dep("//xla/tsl/platform/default:statusor.h"),
        clean_dep("//xla/tsl/platform/default:tracing_impl.h"),
        clean_dep("//xla/tsl/platform/default:unbounded_work_queue.h"),
    ] + select({
        clean_dep("@local_xla//xla/tsl:windows"): [
            clean_dep("//xla/tsl/platform/windows:intrinsics_port.h"),
            clean_dep("//xla/tsl/platform/windows:stacktrace.h"),
            clean_dep("//xla/tsl/platform/windows:subprocess.h"),
            clean_dep("//xla/tsl/platform/windows:wide_char.h"),
            clean_dep("//xla/tsl/platform/windows:windows_file_system.h"),
        ],
        "//conditions:default": [
            clean_dep("//xla/tsl/platform/default:posix_file_system.h"),
            clean_dep("//xla/tsl/platform/default:subprocess.h"),
        ],
    })

def tf_additional_all_protos():
    return ["//tensorflow/core:protos_all"]

def tf_protos_profiler_service():
    return [
        clean_dep("//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl"),
        clean_dep("//tsl/profiler/protobuf:profiler_service_proto_cc_impl"),
        clean_dep("//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl"),
    ]

# TODO(jakeharmon): Move TSL macros that reference TF targets back into TF
def tf_protos_grappler_impl():
    return ["//tensorflow/core/grappler/costs:op_performance_data_cc_impl"]

def tf_protos_grappler():
    return if_static(
        extra_deps = tf_protos_grappler_impl(),
        otherwise = ["//tensorflow/core/grappler/costs:op_performance_data_cc"],
    )

def tf_additional_device_tracer_srcs():
    return [
        "device_tracer_cuda.cc",
        "device_tracer_rocm.cc",
    ]

def tf_additional_test_deps():
    return []

def tf_additional_lib_deps():
    """Additional dependencies needed to build TF libraries."""
    return [
        "@com_google_absl//absl/base:base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ]

def tf_additional_core_deps():
    return select({
        clean_dep("@local_xla//xla/tsl:android"): [],
        clean_dep("@local_xla//xla/tsl:ios"): [],
        clean_dep("@local_xla//xla/tsl:linux_s390x"): [],
        "//conditions:default": [
            clean_dep("@local_xla//xla/tsl/platform/cloud:gcs_file_system"),
        ],
    })

def tf_lib_proto_parsing_deps():
    return [
        ":protos_all_cc",
        clean_dep("@eigen_archive//:eigen3"),
        clean_dep("@local_xla//xla/tsl/protobuf:protos_all_cc"),
    ]

def tf_py_clif_cc(name, visibility = None, **_kwargs):
    _ = visibility  # @unused
    pass

def tf_pyclif_proto_library(
        name,
        proto_lib,
        proto_srcfile = "",
        visibility = None,
        **_kwargs):
    _ = (proto_lib, proto_srcfile, visibility)  # @unused
    native.filegroup(name = name)
    native.filegroup(name = name + "_pb2")

def tf_additional_rpc_deps():
    return []

def tf_additional_tensor_coding_deps():
    return []

def tf_fingerprint_deps():
    return [
        "@farmhash_archive//:farmhash",
    ]

def tf_protobuf_deps():
    return if_static(
        [
            clean_dep("@com_google_protobuf//:protobuf"),
        ],
        otherwise = [clean_dep("@com_google_protobuf//:protobuf_headers")],
    )

# TODO(b/356020232): remove completely after migration is done
# Link protobuf, unless the tsl_link_protobuf build flag is explicitly set to false.
def tsl_protobuf_deps():
    return if_tsl_link_protobuf([clean_dep("@com_google_protobuf//:protobuf")], [clean_dep("@com_google_protobuf//:protobuf_headers")])

def strict_cc_test(
        name,
        linkstatic = True,
        shuffle_tests = True,
        args = None,
        fail_if_no_test_linked = True,
        fail_if_no_test_selected = True,
        **kwargs):
    """A drop-in replacement for cc_test that enforces some good practices by default.

    This should be lightweight and not add any dependencies by itself.

    Args:
      name: The name of the test.
      linkstatic: Whether to link statically.
      shuffle_tests: Whether to shuffle the test cases.
      args: The arguments to pass to the test.
      fail_if_no_test_linked: Whether to fail if no tests are linked.
      fail_if_no_test_selected: Whether to fail if no tests are selected to run.
      **kwargs: Other arguments to pass to the test.
    """

    if args == None:
        args = []

    if shuffle_tests:
        # Shuffle tests to avoid test ordering dependencies.
        args = args + ["--gtest_shuffle"]

    if fail_if_no_test_linked:
        # Fail if no tests are linked. This is to avoid having a test target that does not run any
        # tests. This can happen if the test's link options are not set correctly.
        args = args + ["--gtest_fail_if_no_test_linked"]

    if fail_if_no_test_selected:
        # Fail if no tests are selected. This is to avoid having a test target that does not run any
        # tests. This can happen if the test has extraneous shards or disables all its test cases.
        args.append("--gtest_fail_if_no_test_selected")

    native.cc_test(
        name = name,
        linkstatic = linkstatic,
        args = args,
        **kwargs
    )

# When tsl_protobuf_header_only is true, we need to add the protobuf library
# back into our binaries explicitly.
def tsl_cc_test(
        name,
        deps = [],
        **kwargs):
    """A wrapper around strict_cc_test that adds protobuf deps if needed.

    It also defaults to linkstatic = True, which is a good practice for catching duplicate
    symbols at link time (e.g. linking in two main() functions).

    By default, it also shuffles the tests to avoid test ordering dependencies.

    Use tsl_cc_test instead of cc_test in all .../tsl/... directories.

    Args:
      name: The name of the test.
      deps: The dependencies of the test.
      **kwargs: Other arguments to pass to the test.
    """

    strict_cc_test(
        name = name,
        deps = deps + if_tsl_link_protobuf(
            [],
            [
                clean_dep("@com_google_protobuf//:protobuf"),
                # TODO(ddunleavy) remove these and add proto deps to tests
                # granularly
                clean_dep("@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc_impl"),
                clean_dep("@local_xla//xla/tsl/protobuf:histogram_proto_cc_impl"),
                clean_dep("@local_xla//xla/tsl/protobuf:status_proto_cc_impl"),
                clean_dep("//tsl/profiler/protobuf:xplane_proto_cc_impl"),
                clean_dep("//tsl/profiler/protobuf:profiler_options_proto_cc_impl"),
            ],
        ),
        **kwargs
    )

def tf_portable_proto_lib():
    return ["//tensorflow/core:protos_all_cc_impl", clean_dep("@local_xla//xla/tsl/protobuf:protos_all_cc_impl")]

def tf_protobuf_compiler_deps():
    return if_static(
        [
            clean_dep("@com_google_protobuf//:protobuf"),
        ],
        otherwise = [clean_dep("@com_google_protobuf//:protobuf_headers")],
    )

def tf_windows_aware_platform_deps(name):
    return select({
        clean_dep("@local_xla//xla/tsl:windows"): [
            clean_dep("@local_xla//xla/tsl/platform/windows:" + name),
        ],
        "//conditions:default": [
            clean_dep("//xla/tsl/platform/default:" + name),
        ],
    })

def tf_platform_deps(name, platform_dir = "@local_xla//xla/tsl/platform/"):
    return [platform_dir + "default:" + name]

def tf_stream_executor_deps(name, platform_dir = "@local_xla//xla/tsl/platform/"):
    return tf_platform_deps(name, platform_dir)

def tf_platform_alias(name, platform_dir = "@local_xla//xla/tsl/platform/"):
    return [platform_dir + "default:" + name]

def tf_logging_deps():
    return [clean_dep("//xla/tsl/platform/default:logging")]

def tf_error_logging_deps():
    return [clean_dep("//xla/tsl/platform/default:error_logging")]

def tsl_grpc_credentials_deps():
    return [clean_dep("//xla/tsl/platform/default:grpc_credentials")]

def tf_resource_deps():
    return [clean_dep("//xla/tsl/platform/default:resource")]

def tf_portable_deps_no_runtime():
    return [
        "@eigen_archive//:eigen3",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
    ]

def tf_google_mobile_srcs_no_runtime():
    return []

def tf_google_mobile_srcs_only_runtime():
    return []

def tf_cuda_root_path_deps():
    return tf_platform_deps("cuda_root_path")
