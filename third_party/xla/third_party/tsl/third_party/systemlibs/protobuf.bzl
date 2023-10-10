def _GetPath(ctx, path):
    if ctx.label.workspace_root:
        return ctx.label.workspace_root + "/" + path
    else:
        return path

def _IsNewExternal(ctx):
    # Bazel 0.4.4 and older have genfiles paths that look like:
    #   bazel-out/local-fastbuild/genfiles/external/repo/foo
    # After the exec root rearrangement, they look like:
    #   ../repo/bazel-out/local-fastbuild/genfiles/foo
    return ctx.label.workspace_root.startswith("../")

def _GenDir(ctx):
    if _IsNewExternal(ctx):
        # We are using the fact that Bazel 0.4.4+ provides repository-relative paths
        # for ctx.genfiles_dir.
        return ctx.genfiles_dir.path + (
            "/" + ctx.attr.includes[0] if ctx.attr.includes and ctx.attr.includes[0] else ""
        )

    # This means that we're either in the old version OR the new version in the local repo.
    # Either way, appending the source path to the genfiles dir works.
    return ctx.var["GENDIR"] + "/" + _SourceDir(ctx)

def _SourceDir(ctx):
    if not ctx.attr.includes:
        return ctx.label.workspace_root
    if not ctx.attr.includes[0]:
        return _GetPath(ctx, ctx.label.package)
    if not ctx.label.package:
        return _GetPath(ctx, ctx.attr.includes[0])
    return _GetPath(ctx, ctx.label.package + "/" + ctx.attr.includes[0])

def _CcHdrs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
    return ret

def _CcSrcs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
    return ret

def _CcOuts(srcs, use_grpc_plugin = False):
    return _CcHdrs(srcs, use_grpc_plugin) + _CcSrcs(srcs, use_grpc_plugin)

def _PyOuts(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + "_pb2.py" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + "_pb2_grpc.py" for s in srcs]
    return ret

def _RelativeOutputPath(path, include, dest = ""):
    if include == None:
        return path

    if not path.startswith(include):
        fail("Include path %s isn't part of the path %s." % (include, path))

    if include and include[-1] != "/":
        include = include + "/"
    if dest and dest[-1] != "/":
        dest = dest + "/"

    path = path[len(include):]
    return dest + path

def _proto_gen_impl(ctx):
    """General implementation for generating protos"""
    srcs = ctx.files.srcs
    deps = []
    deps += ctx.files.srcs
    source_dir = _SourceDir(ctx)
    gen_dir = _GenDir(ctx)
    if source_dir:
        import_flags = ["-I" + source_dir, "-I" + gen_dir]
    else:
        import_flags = ["-I."]

    for dep in ctx.attr.deps:
        import_flags += dep.proto.import_flags
        deps += dep.proto.deps
    import_flags = depset(import_flags).to_list()
    deps = depset(deps).to_list()

    args = []
    if ctx.attr.gen_cc:
        args += ["--cpp_out=" + gen_dir]
    if ctx.attr.gen_py:
        args += ["--python_out=" + gen_dir]

    inputs = srcs + deps
    tools = [ctx.executable.protoc]
    if ctx.executable.plugin:
        plugin = ctx.executable.plugin
        lang = ctx.attr.plugin_language
        if not lang and plugin.basename.startswith("protoc-gen-"):
            lang = plugin.basename[len("protoc-gen-"):]
        if not lang:
            fail("cannot infer the target language of plugin", "plugin_language")

        outdir = gen_dir
        if ctx.attr.plugin_options:
            outdir = ",".join(ctx.attr.plugin_options) + ":" + outdir
        args += ["--plugin=protoc-gen-%s=%s" % (lang, plugin.path)]
        args += ["--%s_out=%s" % (lang, outdir)]
        tools.append(plugin)

    if args:
        ctx.actions.run(
            inputs = inputs,
            outputs = ctx.outputs.outs,
            arguments = args + import_flags + [s.path for s in srcs],
            executable = ctx.executable.protoc,
            mnemonic = "ProtoCompile",
            tools = tools,
            use_default_shell_env = True,
        )

    return struct(
        proto = struct(
            srcs = srcs,
            import_flags = import_flags,
            deps = deps,
        ),
    )

proto_gen = rule(
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "deps": attr.label_list(providers = ["proto"]),
        "includes": attr.string_list(),
        "protoc": attr.label(
            cfg = "host",
            executable = True,
            allow_single_file = True,
            mandatory = True,
        ),
        "plugin": attr.label(
            cfg = "host",
            allow_files = True,
            executable = True,
        ),
        "plugin_language": attr.string(),
        "plugin_options": attr.string_list(),
        "gen_cc": attr.bool(),
        "gen_py": attr.bool(),
        "outs": attr.output_list(),
    },
    output_to_genfiles = True,
    implementation = _proto_gen_impl,
)
"""Generates codes from Protocol Buffers definitions.

This rule helps you to implement Skylark macros specific to the target
language. You should prefer more specific `cc_proto_library `,
`py_proto_library` and others unless you are adding such wrapper macros.

Args:
  srcs: Protocol Buffers definition files (.proto) to run the protocol compiler
    against.
  deps: a list of dependency labels; must be other proto libraries.
  includes: a list of include paths to .proto files.
  protoc: the label of the protocol compiler to generate the sources.
  plugin: the label of the protocol compiler plugin to be passed to the protocol
    compiler.
  plugin_language: the language of the generated sources
  plugin_options: a list of options to be passed to the plugin
  gen_cc: generates C++ sources in addition to the ones from the plugin.
  gen_py: generates Python sources in addition to the ones from the plugin.
  outs: a list of labels of the expected outputs from the protocol compiler.
"""

def cc_proto_library(
        name,
        srcs = [],
        deps = [],
        cc_libs = [],
        include = None,
        protoc = "@com_google_protobuf//:protoc",
        internal_bootstrap_hack = False,
        use_grpc_plugin = False,
        default_runtime = "@com_google_protobuf//:protobuf",
        **kwargs):
    """Bazel rule to create a C++ protobuf library from proto source files

    NOTE: the rule is only an internal workaround to generate protos. The
    interface may change and the rule may be removed when bazel has introduced
    the native rule.

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
      default_runtime: the implicitly default runtime which will be depended on by
          the generated cc_library target.
      **kwargs: other keyword arguments that are passed to cc_library.

    """

    includes = []
    if include != None:
        includes = [include]

    if internal_bootstrap_hack:
        # For pre-checked-in generated files, we add the internal_bootstrap_hack
        # which will skip the codegen action.
        proto_gen(
            name = name + "_genproto",
            srcs = srcs,
            deps = [s + "_genproto" for s in deps],
            includes = includes,
            protoc = protoc,
            visibility = ["//visibility:public"],
        )

        # An empty cc_library to make rule dependency consistent.
        native.cc_library(
            name = name,
            **kwargs
        )
        return

    grpc_cpp_plugin = None
    if use_grpc_plugin:
        grpc_cpp_plugin = "//external:grpc_cpp_plugin"

    gen_srcs = _CcSrcs(srcs, use_grpc_plugin)
    gen_hdrs = _CcHdrs(srcs, use_grpc_plugin)
    outs = gen_srcs + gen_hdrs

    proto_gen(
        name = name + "_genproto",
        srcs = srcs,
        deps = [s + "_genproto" for s in deps],
        includes = includes,
        protoc = protoc,
        plugin = grpc_cpp_plugin,
        plugin_language = "grpc",
        gen_cc = 1,
        outs = outs,
        visibility = ["//visibility:public"],
    )

    if default_runtime and not default_runtime in cc_libs:
        cc_libs = cc_libs + [default_runtime]
    if use_grpc_plugin:
        cc_libs = cc_libs + ["//external:grpc_lib"]

    native.cc_library(
        name = name,
        srcs = gen_srcs,
        hdrs = gen_hdrs,
        deps = cc_libs + deps,
        includes = includes,
        alwayslink = 1,
        **kwargs
    )

def internal_gen_well_known_protos_java(srcs):
    """Bazel rule to generate the gen_well_known_protos_java genrule

    Args:
      srcs: the well known protos
    """
    root = Label("%s//protobuf_java" % (native.repository_name())).workspace_root
    pkg = native.package_name() + "/" if native.package_name() else ""
    if root == "":
        include = " -I%ssrc " % pkg
    else:
        include = " -I%s/%ssrc " % (root, pkg)
    native.genrule(
        name = "gen_well_known_protos_java",
        srcs = srcs,
        outs = [
            "wellknown.srcjar",
        ],
        cmd = "$(location :protoc) --java_out=$(@D)/wellknown.jar" +
              " %s $(SRCS) " % include +
              " && mv $(@D)/wellknown.jar $(@D)/wellknown.srcjar",
        tools = [":protoc"],
    )

def internal_copied_filegroup(name, srcs, strip_prefix, dest, **kwargs):
    """Macro to copy files to a different directory and then create a filegroup.

    This is used by the //:protobuf_python py_proto_library target to work around
    an issue caused by Python source files that are part of the same Python
    package being in separate directories.

    Args:
      srcs: The source files to copy and add to the filegroup.
      strip_prefix: Path to the root of the files to copy.
      dest: The directory to copy the source files into.
      **kwargs: extra arguments that will be passesd to the filegroup.
    """
    outs = [_RelativeOutputPath(s, strip_prefix, dest) for s in srcs]

    native.genrule(
        name = name + "_genrule",
        srcs = srcs,
        outs = outs,
        cmd = " && ".join(
            ["cp $(location %s) $(location %s)" %
             (s, _RelativeOutputPath(s, strip_prefix, dest)) for s in srcs],
        ),
    )

    native.filegroup(
        name = name,
        srcs = outs,
        **kwargs
    )

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
    outs = _PyOuts(srcs, use_grpc_plugin)

    includes = []
    if include != None:
        includes = [include]

    grpc_python_plugin = None
    if use_grpc_plugin:
        grpc_python_plugin = "//external:grpc_python_plugin"
        # Note: Generated grpc code depends on Python grpc module. This dependency
        # is not explicitly listed in py_libs. Instead, host system is assumed to
        # have grpc installed.

    proto_gen(
        name = name + "_genproto",
        srcs = srcs,
        deps = [s + "_genproto" for s in deps],
        includes = includes,
        protoc = protoc,
        gen_py = 1,
        outs = outs,
        visibility = ["//visibility:public"],
        plugin = grpc_python_plugin,
        plugin_language = "grpc",
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

def internal_protobuf_py_tests(
        name,
        modules = [],
        **kwargs):
    """Bazel rules to create batch tests for protobuf internal.

    Args:
      name: the name of the rule.
      modules: a list of modules for tests. The macro will create a py_test for
          each of the parameter with the source "google/protobuf/%s.py"
      **kwargs: extra parameters that will be passed into the py_test.

    """
    for m in modules:
        s = "python/google/protobuf/internal/%s.py" % m
        native.py_test(
            name = "py_%s" % m,
            srcs = [s],
            main = s,
            **kwargs
        )

def check_protobuf_required_bazel_version():
    """For WORKSPACE files, to check the installed version of bazel.

    This ensures bazel supports our approach to proto_library() depending on a
    copied filegroup. (Fixed in bazel 0.5.4)
    """
    expected = apple_common.dotted_version("0.5.4")
    current = apple_common.dotted_version(native.bazel_version)
    if current.compare_to(expected) < 0:
        fail("Bazel must be newer than 0.5.4")
