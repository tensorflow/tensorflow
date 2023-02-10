# Platform-specific build configurations.

load("@com_google_protobuf//:protobuf.bzl", "proto_gen")
load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")
load(
    "//tensorflow/tsl:tsl.bzl",
    "clean_dep",
    "if_not_windows",
    "if_tsl_link_protobuf",
)
load("@com_github_grpc_grpc//bazel:generate_cc.bzl", "generate_cc")

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

# Appends a suffix to a list of deps.
def tf_deps(deps, suffix):
    tf_deps = []

    # If the package name is in shorthand form (ie: does not contain a ':'),
    # expand it to the full name.
    for dep in deps:
        tf_dep = dep

        if not ":" in dep:
            dep_pieces = dep.split("/")
            tf_dep += ":" + dep_pieces[len(dep_pieces) - 1]

        tf_deps += [tf_dep + suffix]

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
            deps = cc_deps + ["@org_tensorflow//third_party/python_runtime:headers"],
            linkshared = 1,
            testonly = testonly,
            copts = copts,
        )
        shared_objects.append(shared_object_name)

    # Now create a py_library with these shared objects as data.
    native.py_library(
        name = name,
        srcs = py_srcs,
        deps = py_deps,
        srcs_version = srcs_version,
        data = shared_objects,
        testonly = testonly,
        **kwargs
    )

def _proto_cc_hdrs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
    return ret

def _proto_cc_srcs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
    return ret

def _proto_py_outs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + "_pb2.py" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + "_pb2_grpc.py" for s in srcs]
    return ret

# Re-defined protocol buffer rule to allow building "header only" protocol
# buffers, to avoid duplicate registrations. Also allows non-iterable cc_libs
# containing select() statements.
def cc_proto_library(
        name,
        srcs = [],
        deps = [],
        cc_libs = [],
        include = None,
        protoc = "@com_google_protobuf//:protoc",
        internal_bootstrap_hack = False,
        use_grpc_plugin = False,
        use_grpc_namespace = False,
        make_default_target_header_only = False,
        protolib_name = None,
        protolib_deps = [],
        **kwargs):
    """Bazel rule to create a C++ protobuf library from proto source files.

    Args:
      name: the name of the cc_proto_library.
      srcs: the .proto files of the cc_proto_library.
      deps: a list of dependency labels; must be cc_proto_library.
      cc_libs: a list of other cc_library targets depended by the generated
          cc_library.
      include: a string indicating the include path of the .proto files.
      protoc: the label of the protocol compiler to generate the sources.
      internal_bootstrap_hack: a flag indicating if the cc_proto_library is used only
          for bootstrapping. When it is set to True, no files will be generated.
          The rule will simply be a provider for .proto files, so that other
          cc_proto_library can depend on it.
      use_grpc_plugin: a flag to indicate whether to call the grpc C++ plugin
          when processing the proto files.
      use_grpc_namespace: the namespace for the grpc services.
      make_default_target_header_only: Controls the naming of generated
          rules. If True, the `name` rule will be header-only, and an _impl rule
          will contain the implementation. Otherwise the header-only rule (name
          + "_headers_only") must be referred to explicitly.
      protolib_name: the name for the proto library generated by this rule.
      protolib_deps: the dependencies to proto libraries.
      **kwargs: other keyword arguments that are passed to cc_library.
    """
    includes = []
    if include != None:
        includes = [include]
    if protolib_name == None:
        protolib_name = name

    genproto_deps = ([s + "_genproto" for s in protolib_deps] +
                     ["@com_google_protobuf//:cc_wkt_protos_genproto"])
    if internal_bootstrap_hack:
        # For pre-checked-in generated files, we add the internal_bootstrap_hack
        # which will skip the codegen action.
        proto_gen(
            name = protolib_name + "_genproto",
            srcs = srcs,
            includes = includes,
            protoc = protoc,
            visibility = ["//visibility:public"],
            deps = genproto_deps,
        )

        # An empty cc_library to make rule dependency consistent.
        native.cc_library(
            name = name,
            **kwargs
        )
        return

    grpc_cpp_plugin = None
    plugin_options = []
    if use_grpc_plugin:
        grpc_cpp_plugin = "//external:grpc_cpp_plugin"
        if use_grpc_namespace:
            plugin_options = ["services_namespace=grpc"]

    gen_srcs = _proto_cc_srcs(srcs, use_grpc_plugin)
    gen_hdrs = _proto_cc_hdrs(srcs, use_grpc_plugin)
    outs = gen_srcs + gen_hdrs

    proto_gen(
        name = protolib_name + "_genproto",
        srcs = srcs,
        outs = outs,
        gen_cc = 1,
        includes = includes,
        plugin = grpc_cpp_plugin,
        plugin_language = "grpc",
        plugin_options = plugin_options,
        protoc = protoc,
        visibility = ["//visibility:public"],
        deps = genproto_deps,
    )

    if use_grpc_plugin:
        cc_libs += select({
            clean_dep("//tensorflow/tsl:linux_s390x"): ["//external:grpc_lib_unsecure"],
            "//conditions:default": ["//external:grpc_lib"],
        })

    impl_name = name + "_impl"
    header_only_name = name + "_headers_only"
    header_only_deps = tf_deps(protolib_deps, "_cc_headers_only")

    if make_default_target_header_only:
        native.alias(
            name = name,
            actual = header_only_name,
            visibility = kwargs["visibility"],
        )
    else:
        native.alias(
            name = name,
            actual = impl_name,
            visibility = kwargs["visibility"],
        )

    native.cc_library(
        name = impl_name,
        srcs = gen_srcs,
        hdrs = gen_hdrs,
        deps = cc_libs + deps,
        includes = includes,
        alwayslink = 1,
        **kwargs
    )
    native.cc_library(
        name = header_only_name,
        deps = [
            "@com_google_protobuf//:protobuf_headers",
        ] + header_only_deps + if_tsl_link_protobuf([impl_name]),
        hdrs = gen_hdrs,
        **kwargs
    )

# Re-defined protocol buffer rule to allow setting service namespace.
def cc_grpc_library(
        name,
        srcs,
        deps,
        well_known_protos = False,
        generate_mocks = False,
        service_namespace = "grpc",
        **kwargs):
    """Generates C++ grpc classes for services defined in a proto file.

    This rule is compatible with proto_library and
    cc_proto_library native rules such that it expects proto_library target
    as srcs argument and generates only grpc library classes, expecting
    protobuf messages classes library (cc_proto_library target) to be passed in
    deps argument.
    Assumes the generated classes will be used in cc_api_version = 2.
    Args:
        name (str): Name of rule.
        srcs (list): A single .proto file which contains services definitions,
          or if grpc_only parameter is True, a single proto_library which
          contains services descriptors.
        deps (list): A list of C++ proto_library (or cc_proto_library) which
          provides the compiled code of any message that the services depend on.
        well_known_protos (bool): Should this library additionally depend on
          well known protos. Deprecated, the well known protos should be
          specified as explicit dependencies of the proto_library target
          (passed in srcs parameter) instead. False by default.
        generate_mocks (bool): when True, Google Mock code for client stub is
          generated. False by default.
        service_namespace (str): Service namespace.
        **kwargs: rest of arguments, e.g., compatible_with and visibility
    """
    if len(srcs) > 1:
        fail("Only one srcs value supported", "srcs")

    extra_deps = []
    proto_targets = []

    if not srcs:
        fail("srcs cannot be empty", "srcs")
    proto_targets += srcs

    extra_deps += select({
        clean_dep("//tensorflow/tsl:linux_s390x"): ["//external:grpc_lib_unsecure"],
        "//conditions:default": ["//external:grpc_lib"],
    })

    codegen_grpc_target = "_" + name + "_grpc_codegen"
    generate_cc(
        name = codegen_grpc_target,
        srcs = proto_targets,
        plugin = "//external:grpc_cpp_plugin",
        well_known_protos = well_known_protos,
        generate_mocks = generate_mocks,
        flags = ["services_namespace=" + service_namespace],
        **kwargs
    )

    native.cc_library(
        name = name,
        srcs = [":" + codegen_grpc_target],
        hdrs = [":" + codegen_grpc_target],
        deps = deps +
               extra_deps,
        **kwargs
    )

# Re-defined protocol buffer rule to bring in the change introduced in commit
# https://github.com/google/protobuf/commit/294b5758c373cbab4b72f35f4cb62dc1d8332b68
# which was not part of a stable protobuf release in 04/2018.
# TODO(jsimsa): Remove this once the protobuf dependency version is updated
# to include the above commit.
def py_proto_library(
        name,
        srcs = [],
        deps = [],
        py_libs = [],
        py_extra_srcs = [],
        include = None,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        use_grpc_plugin = False,
        **kwargs):
    """Bazel rule to create a Python protobuf library from proto source files

    NOTE: the rule is only an internal workaround to generate protos. The
    interface may change and the rule may be removed when bazel has introduced
    the native rule.

    Args:
      name: the name of the py_proto_library.
      srcs: the .proto files of the py_proto_library.
      deps: a list of dependency labels; must be py_proto_library.
      py_libs: a list of other py_library targets depended by the generated
          py_library.
      py_extra_srcs: extra source files that will be added to the output
          py_library. This attribute is used for internal bootstrapping.
      include: a string indicating the include path of the .proto files.
      default_runtime: the implicitly default runtime which will be depended on by
          the generated py_library target.
      protoc: the label of the protocol compiler to generate the sources.
      use_grpc_plugin: a flag to indicate whether to call the Python C++ plugin
          when processing the proto files.
      **kwargs: other keyword arguments that are passed to py_library.
    """
    outs = _proto_py_outs(srcs, use_grpc_plugin)

    includes = []
    if include != None:
        includes = [include]

    grpc_python_plugin = None
    if use_grpc_plugin:
        grpc_python_plugin = "//external:grpc_python_plugin"
        # Note: Generated grpc code depends on Python grpc module. This dependency
        # is not explicitly listed in py_libs. Instead, host system is assumed to
        # have grpc installed.

    genproto_deps = []
    for dep in deps:
        if dep != "@com_google_protobuf//:protobuf_python":
            genproto_deps.append(dep + "_genproto")
        else:
            genproto_deps.append("@com_google_protobuf//:well_known_types_py_pb2_genproto")

    proto_gen(
        name = name + "_genproto",
        srcs = srcs,
        outs = outs,
        gen_py = 1,
        includes = includes,
        plugin = grpc_python_plugin,
        plugin_language = "grpc",
        protoc = protoc,
        visibility = ["//visibility:public"],
        deps = genproto_deps,
    )

    if default_runtime and not default_runtime in py_libs + deps:
        py_libs = py_libs + [default_runtime]

    native.py_library(
        name = name,
        srcs = outs + py_extra_srcs,
        deps = py_libs + deps,
        imports = includes,
        **kwargs
    )

def tf_proto_library_cc(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = None,
        testonly = 0,
        cc_libs = [],
        cc_stubby_versions = None,
        cc_grpc_version = None,
        use_grpc_namespace = False,
        j2objc_api_version = 1,
        cc_api_version = 2,
        js_codegen = "jspb",
        create_service = False,
        create_java_proto = False,
        make_default_target_header_only = False):
    js_codegen = js_codegen  # unused argument
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
        testonly = testonly,
        visibility = visibility,
    )
    _ignore = (create_service, create_java_proto)

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True

    protolib_deps = tf_deps(protodeps, "")
    cc_deps = tf_deps(protodeps, "_cc")
    cc_name = name + "_cc"
    if not srcs:
        # This is a collection of sub-libraries. Build header-only and impl
        # libraries containing all the sources.
        proto_gen(
            name = name + "_genproto",
            protoc = "@com_google_protobuf//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in protolib_deps],
        )

        native.alias(
            name = cc_name + "_genproto",
            actual = name + "_genproto",
            testonly = testonly,
            visibility = visibility,
        )

        native.alias(
            name = cc_name + "_headers_only",
            actual = cc_name,
            testonly = testonly,
            visibility = visibility,
        )

        native.cc_library(
            name = cc_name,
            deps = cc_deps + ["@com_google_protobuf//:protobuf_headers"] + if_tsl_link_protobuf([name + "_cc_impl"]),
            testonly = testonly,
            visibility = visibility,
        )
        native.cc_library(
            name = cc_name + "_impl",
            deps = [s + "_impl" for s in cc_deps],
        )

        return

    cc_proto_library(
        name = cc_name,
        protolib_name = name,
        testonly = testonly,
        srcs = srcs,
        cc_libs = cc_libs + if_tsl_link_protobuf(
            ["@com_google_protobuf//:protobuf"],
            ["@com_google_protobuf//:protobuf_headers"],
        ),
        copts = if_not_windows([
            "-Wno-unknown-warning-option",
            "-Wno-unused-but-set-variable",
            "-Wno-sign-compare",
        ]),
        make_default_target_header_only = make_default_target_header_only,
        protoc = "@com_google_protobuf//:protoc",
        use_grpc_plugin = use_grpc_plugin,
        use_grpc_namespace = use_grpc_namespace,
        visibility = visibility,
        deps = cc_deps,
        protolib_deps = protolib_deps,
    )

def tf_proto_library_py(
        name,
        srcs = [],
        protodeps = [],
        deps = [],
        visibility = None,
        testonly = 0,
        srcs_version = "PY3",
        use_grpc_plugin = False):
    py_deps = tf_deps(protodeps, "_py")
    py_name = name + "_py"
    if not srcs:
        # This is a collection of sub-libraries. Build header-only and impl
        # libraries containing all the sources.
        proto_gen(
            name = py_name + "_genproto",
            protoc = "@com_google_protobuf//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in py_deps],
        )
        native.py_library(
            name = py_name,
            deps = py_deps + [clean_dep("@com_google_protobuf//:protobuf_python")],
            testonly = testonly,
            visibility = visibility,
        )
        return

    py_proto_library(
        name = py_name,
        testonly = testonly,
        srcs = srcs,
        default_runtime = clean_dep("@com_google_protobuf//:protobuf_python"),
        protoc = "@com_google_protobuf//:protoc",
        srcs_version = srcs_version,
        use_grpc_plugin = use_grpc_plugin,
        visibility = visibility,
        deps = deps + py_deps + [clean_dep("@com_google_protobuf//:protobuf_python")],
    )

def tf_jspb_proto_library(**kwargs):
    pass

def tf_proto_library(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = None,
        testonly = 0,
        cc_libs = [],
        cc_stubby_versions = None,
        cc_api_version = 2,
        cc_grpc_version = None,
        use_grpc_namespace = False,
        j2objc_api_version = 1,
        js_codegen = "jspb",
        create_service = False,
        create_java_proto = False,
        create_go_proto = False,
        create_grpc_library = False,
        make_default_target_header_only = False,
        exports = [],
        tags = []):
    """Make a proto library, possibly depending on other proto libraries."""

    # TODO(b/145545130): Add docstring explaining what rules this creates and how
    # opensource projects importing TF in bazel can use them safely (i.e. w/o ODR or
    # ABI violations).
    _ignore = (
        js_codegen,
        create_service,
        create_java_proto,
        cc_stubby_versions,
        create_go_proto,
    )

    if name.endswith("_proto"):
        name_sans_proto = name[:-6]
    else:
        name_sans_proto = name

    native.proto_library(
        name = name,
        srcs = srcs,
        deps = protodeps + well_known_proto_libs(),
        exports = exports,
        visibility = visibility,
        testonly = testonly,
        tags = tags,
    )

    tf_proto_library_cc(
        name = name,
        testonly = testonly,
        srcs = srcs,
        cc_grpc_version = cc_grpc_version,
        use_grpc_namespace = use_grpc_namespace,
        cc_libs = cc_libs,
        make_default_target_header_only = make_default_target_header_only,
        protodeps = protodeps,
        visibility = visibility,
    )

    if create_grpc_library:
        cc_grpc_library(
            name = name_sans_proto + "_cc_grpc_proto",
            srcs = [name],
            generate_mocks = True,
            deps = [name + "_cc"],
            visibility = visibility,
            testonly = testonly,
        )

    tf_proto_library_py(
        name = name,
        testonly = testonly,
        srcs = srcs,
        protodeps = protodeps,
        srcs_version = "PY3",
        use_grpc_plugin = has_services,
        visibility = visibility,
    )

def tf_additional_lib_hdrs():
    return [
        clean_dep("//tensorflow/tsl/platform/default:casts.h"),
        clean_dep("//tensorflow/tsl/platform/default:context.h"),
        clean_dep("//tensorflow/tsl/platform/default:cord.h"),
        clean_dep("//tensorflow/tsl/platform/default:dynamic_annotations.h"),
        clean_dep("//tensorflow/tsl/platform/default:integral_types.h"),
        clean_dep("//tensorflow/tsl/platform/default:logging.h"),
        clean_dep("//tensorflow/tsl/platform/default:mutex.h"),
        clean_dep("//tensorflow/tsl/platform/default:mutex_data.h"),
        clean_dep("//tensorflow/tsl/platform/default:notification.h"),
        clean_dep("//tensorflow/tsl/platform/default:stacktrace.h"),
        clean_dep("//tensorflow/tsl/platform/default:status.h"),
        clean_dep("//tensorflow/tsl/platform/default:tracing_impl.h"),
        clean_dep("//tensorflow/tsl/platform/default:unbounded_work_queue.h"),
    ] + select({
        clean_dep("//tensorflow/tsl:windows"): [
            clean_dep("//tensorflow/tsl/platform/windows:intrinsics_port.h"),
            clean_dep("//tensorflow/tsl/platform/windows:stacktrace.h"),
            clean_dep("//tensorflow/tsl/platform/windows:subprocess.h"),
            clean_dep("//tensorflow/tsl/platform/windows:wide_char.h"),
            clean_dep("//tensorflow/tsl/platform/windows:windows_file_system.h"),
        ],
        "//conditions:default": [
            clean_dep("//tensorflow/tsl/platform/default:posix_file_system.h"),
            clean_dep("//tensorflow/tsl/platform/default:subprocess.h"),
        ],
    })

def tf_additional_all_protos():
    return [clean_dep("//tensorflow/core:protos_all")]

def tf_protos_all():
    return if_static(
        extra_deps = [
            clean_dep("//tensorflow/core/protobuf:conv_autotuning_proto_cc_impl"),
            clean_dep("//tensorflow/core:protos_all_cc_impl"),
            clean_dep("//tensorflow/tsl/protobuf:autotuning_proto_cc_impl"),
            clean_dep("//tensorflow/tsl/protobuf:protos_all_cc_impl"),
        ],
        otherwise = [clean_dep("//tensorflow/core:protos_all_cc")],
    )

def tf_protos_profiler_service():
    return [
        clean_dep("//tensorflow/tsl/profiler/protobuf:profiler_analysis_proto_cc_impl"),
        clean_dep("//tensorflow/tsl/profiler/protobuf:profiler_service_proto_cc_impl"),
        clean_dep("//tensorflow/tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl"),
    ]

def tf_protos_grappler_impl():
    return [clean_dep("//tensorflow/core/grappler/costs:op_performance_data_cc_impl")]

def tf_protos_grappler():
    return if_static(
        extra_deps = tf_protos_grappler_impl(),
        otherwise = [clean_dep("//tensorflow/core/grappler/costs:op_performance_data_cc")],
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
        "@com_google_absl//absl/types:optional",
    ] + if_static(
        [clean_dep("@nsync//:nsync_cpp")],
        [clean_dep("@nsync//:nsync_headers")],
    )

def tf_additional_core_deps():
    return select({
        clean_dep("//tensorflow/tsl:android"): [],
        clean_dep("//tensorflow/tsl:ios"): [],
        clean_dep("//tensorflow/tsl:linux_s390x"): [],
        "//conditions:default": [
            clean_dep("//tensorflow/tsl/platform/cloud:gcs_file_system"),
        ],
    })

def tf_lib_proto_parsing_deps():
    return [
        ":protos_all_cc",
        clean_dep("//third_party/eigen3"),
        clean_dep("//tensorflow/tsl/platform/default/build_config:proto_parsing"),
    ]

def tf_py_clif_cc(name, visibility = None, **kwargs):
    pass

def tf_pyclif_proto_library(
        name,
        proto_lib,
        proto_srcfile = "",
        visibility = None,
        **kwargs):
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

# Link protobuf, unless the tsl_link_protobuf build flag is explicitly set to false.
def tsl_protobuf_deps():
    return if_tsl_link_protobuf([clean_dep("@com_google_protobuf//:protobuf")], [clean_dep("@com_google_protobuf//:protobuf_headers")])

# When tsl_protobuf_header_only is true, we need to add the protobuf library
# back into our binaries explicitly.
def tsl_cc_test(
        name,
        deps = [],
        **kwargs):
    native.cc_test(
        name = name,
        deps = deps + if_tsl_link_protobuf(
            [],
            [
                clean_dep("@com_google_protobuf//:protobuf"),
                # TODO(ddunleavy) remove these and add proto deps to tests
                # granularly
                clean_dep("//tensorflow/tsl/protobuf:error_codes_proto_impl_cc_impl"),
                clean_dep("//tensorflow/tsl/protobuf:histogram_proto_cc_impl"),
                clean_dep("//tensorflow/tsl/protobuf:status_proto_cc_impl"),
                clean_dep("//tensorflow/tsl/profiler/protobuf:xplane_proto_cc_impl"),
                clean_dep("//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc_impl"),
            ],
        ),
        **kwargs
    )

def tf_portable_proto_lib():
    return ["//tensorflow/core:protos_all_cc_impl"]

def tf_protobuf_compiler_deps():
    return if_static(
        [
            clean_dep("@com_google_protobuf//:protobuf"),
        ],
        otherwise = [clean_dep("@com_google_protobuf//:protobuf_headers")],
    )

def tf_windows_aware_platform_deps(name):
    return select({
        "//tensorflow/tsl:windows": [
            "//tensorflow/tsl/platform/windows:" + name,
        ],
        "//conditions:default": [
            "//tensorflow/tsl/platform/default:" + name,
        ],
    })

def tf_platform_deps(name, platform_dir = "//tensorflow/tsl/platform/"):
    return [platform_dir + "default:" + name]

def tf_testing_deps(name, platform_dir = "//tensorflow/tsl/platform/"):
    return tf_platform_deps(name, platform_dir)

def tf_stream_executor_deps(name, platform_dir = "//tensorflow/tsl/platform/"):
    return tf_platform_deps(name, platform_dir)

def tf_platform_alias(name, platform_dir = "//tensorflow/tsl/platform/"):
    return [platform_dir + "default:" + name]

def tf_logging_deps():
    return [clean_dep("//tensorflow/tsl/platform/default:logging")]

def tf_resource_deps():
    return [clean_dep("//tensorflow/tsl/platform/default:resource")]

def tf_portable_deps_no_runtime():
    return [
        "//third_party/eigen3",
        "@double_conversion//:double-conversion",
        "@nsync//:nsync_cpp",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
    ]

def tf_google_mobile_srcs_no_runtime():
    return []

def tf_google_mobile_srcs_only_runtime():
    return []

def if_llvm_aarch64_available(then, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:linux_aarch64"): then,
        "//conditions:default": otherwise,
    })

def if_llvm_aarch32_available(then, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:linux_armhf"): then,
        "//conditions:default": otherwise,
    })

def if_llvm_arm_available(then, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:linux_aarch64"): then,
        clean_dep("//tensorflow/tsl:linux_armhf"): then,
        "//conditions:default": otherwise,
    })

def if_llvm_powerpc_available(then, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:linux_ppc64le"): then,
        "//conditions:default": otherwise,
    })

def if_llvm_system_z_available(then, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:linux_s390x"): then,
        "//conditions:default": otherwise,
    })

def if_llvm_x86_available(then, otherwise = []):
    return select({
        clean_dep("//tensorflow/tsl:linux_x86_64"): then,
        "//conditions:default": otherwise,
    })

def tf_cuda_libdevice_path_deps():
    return tf_platform_deps("cuda_libdevice_path")
