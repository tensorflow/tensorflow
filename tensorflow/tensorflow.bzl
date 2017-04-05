# -*- Python -*-


# Given a source file, generate a test name.
# i.e. "common_runtime/direct_session_test.cc" becomes
#      "common_runtime_direct_session_test"
def src_to_test_name(src):
  return src.replace("/", "_").split(".")[0]


# Return the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_sycl_tests_tags",
    "tf_additional_xla_deps_py",)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "cuda_default_copts")

load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",)


# List of proto files for android builds
def tf_android_core_proto_sources(core_proto_sources_relative):
  return [
      "//tensorflow/core:" + p for p in core_proto_sources_relative
  ]


# Returns the list of pb.h and proto.h headers that are generated for
# tf_android_core_proto_sources().
def tf_android_core_proto_headers(core_proto_sources_relative):
  return ([
      "//tensorflow/core/" + p.replace(".proto", ".pb.h")
      for p in core_proto_sources_relative
  ] + [
      "//tensorflow/core/" + p.replace(".proto", ".proto.h")
      for p in core_proto_sources_relative
  ])


def if_android_x86(a):
  return select({
      str(Label("//tensorflow:android_x86")): a,
      str(Label("//tensorflow:android_x86_64")): a,
      "//conditions:default": [],
  })


def if_android_arm(a):
  return select({
      str(Label("//tensorflow:android_arm")): a,
      "//conditions:default": [],
  })


def if_android_arm64(a):
  return select({
      str(Label("//tensorflow:android_arm64")): a,
      "//conditions:default": [],
  })


def if_not_android(a):
  return select({
      str(Label("//tensorflow:android")): [],
      "//conditions:default": a,
  })


def if_android(a):
  return select({
      str(Label("//tensorflow:android")): a,
      "//conditions:default": [],
  })


def if_ios(a):
  return select({
      str(Label("//tensorflow:ios")): a,
      "//conditions:default": [],
  })


def if_mobile(a):
  return select({
      str(Label("//tensorflow:android")): a,
      str(Label("//tensorflow:ios")): a,
      "//conditions:default": [],
  })


def if_not_mobile(a):
  return select({
      str(Label("//tensorflow:android")): [],
      str(Label("//tensorflow:ios")): [],
      "//conditions:default": a,
  })


def if_not_windows(a):
  return select({
      str(Label("//tensorflow:windows")): [],
      "//conditions:default": a,
  })


def if_x86(a):
  return select({
      str(Label("//tensorflow:linux_x86_64")): a,
      str(Label("//tensorflow:windows")): a,
      "//conditions:default": [],
  })


# LINT.IfChange
def tf_copts():
  return ([
      "-DEIGEN_AVOID_STL_ARRAY",
      "-Iexternal/gemmlowp",
      "-Wno-sign-compare",
      "-fno-exceptions",
  ] + if_cuda(["-DGOOGLE_CUDA=1"]) + if_mkl(["-DINTEL_MKL=1"]) + if_android_arm(
      ["-mfpu=neon"]) + if_x86(["-msse3"]) + select({
          "//tensorflow:android": [
              "-std=c++11",
              "-DTF_LEAN_BINARY",
              "-O2",
          ],
          "//tensorflow:darwin": [],
          "//tensorflow:windows": [
              "/DLANG_CXX11",
              "/D__VERSION__=\\\"MSVC\\\"",
              "/DPLATFORM_WINDOWS",
              "/DTF_COMPILE_LIBRARY",
              "/DEIGEN_HAS_C99_MATH",
              "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
          ],
          "//tensorflow:ios": ["-std=c++11"],
          "//conditions:default": ["-pthread"]
      }))


def tf_opts_nortti_if_android():
  return if_android([
      "-fno-rtti",
      "-DGOOGLE_PROTOBUF_NO_RTTI",
      "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
  ]) + if_android_x86(["-msse4.1"])


# LINT.ThenChange(//tensorflow/contrib/android/cmake/CMakeLists.txt)


# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(op_lib_names, deps=None):
  # Make library out of each op so it can also be used to generate wrappers
  # for various languages.
  if not deps:
    deps = []
  for n in op_lib_names:
    native.cc_library(
        name=n + "_op_lib",
        copts=tf_copts(),
        srcs=["ops/" + n + ".cc"],
        deps=deps + ["//tensorflow/core:framework"],
        visibility=["//visibility:public"],
        alwayslink=1,
        linkstatic=1,)


def tf_gen_op_wrapper_cc(name,
                         out_ops_file,
                         pkg="",
                         op_gen="//tensorflow/cc:cc_op_gen_main",
                         deps=None,
                         override_file=None,
                         include_internal_ops=0):
  # Construct an op generator binary for these ops.
  tool = out_ops_file + "_gen_cc"
  if deps == None:
    deps = [pkg + ":" + name + "_op_lib"]
  native.cc_binary(
      name=tool,
      copts=tf_copts(),
      linkopts=["-lm"],
      linkstatic=1,  # Faster to link this one-time-use binary dynamically
      deps=[op_gen] + deps)

  if override_file == None:
    srcs = []
    override_arg = ","
  else:
    srcs = [override_file]
    override_arg = "$(location " + override_file + ")"
  native.genrule(
      name=name + "_genrule",
      outs=[
          out_ops_file + ".h", out_ops_file + ".cc",
          out_ops_file + "_internal.h", out_ops_file + "_internal.cc"
      ],
      srcs=srcs,
      tools=[":" + tool],
      cmd=("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
           "$(location :" + out_ops_file + ".cc) " + override_arg + " " +
           str(include_internal_ops)))


# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate individual C++ .cc and .h
# files for each of the ops files mentioned, and then generate a
# single cc_library called "name" that combines all the
# generated C++ code.
#
# For example, for:
#  tf_gen_op_wrappers_cc("tf_ops_lib", [ "array_ops", "math_ops" ])
#
#
# This will ultimately generate ops/* files and a library like:
#
# cc_library(name = "tf_ops_lib",
#            srcs = [ "ops/array_ops.cc",
#                     "ops/math_ops.cc" ],
#            hdrs = [ "ops/array_ops.h",
#                     "ops/math_ops.h" ],
#            deps = [ ... ])
#
# Plus a private library for the "hidden" ops.
# cc_library(name = "tf_ops_lib_internal",
#            srcs = [ "ops/array_ops_internal.cc",
#                     "ops/math_ops_internal.cc" ],
#            hdrs = [ "ops/array_ops_internal.h",
#                     "ops/math_ops_internal.h" ],
#            deps = [ ... ])
# TODO(josh11b): Cleaner approach for hidden ops.
def tf_gen_op_wrappers_cc(name,
                          op_lib_names=[],
                          other_srcs=[],
                          other_hdrs=[],
                          pkg="",
                          deps=[
                              str(Label("//tensorflow/cc:ops")),
                              str(Label("//tensorflow/cc:scope")),
                              str(Label("//tensorflow/cc:const_op")),
                          ],
                          op_gen=str(Label("//tensorflow/cc:cc_op_gen_main")),
                          override_file=None,
                          include_internal_ops=0,
                          visibility=None):
  subsrcs = other_srcs
  subhdrs = other_hdrs
  internalsrcs = []
  internalhdrs = []
  for n in op_lib_names:
    tf_gen_op_wrapper_cc(
        n,
        "ops/" + n,
        pkg=pkg,
        op_gen=op_gen,
        override_file=override_file,
        include_internal_ops=include_internal_ops)
    subsrcs += ["ops/" + n + ".cc"]
    subhdrs += ["ops/" + n + ".h"]
    internalsrcs += ["ops/" + n + "_internal.cc"]
    internalhdrs += ["ops/" + n + "_internal.h"]

  native.cc_library(
      name=name,
      srcs=subsrcs,
      hdrs=subhdrs,
      deps=deps + if_not_android([
          "//tensorflow/core:core_cpu",
          "//tensorflow/core:framework",
          "//tensorflow/core:lib",
          "//tensorflow/core:protos_all_cc",
      ]) + if_android([
          "//tensorflow/core:android_tensorflow_lib",
      ]),
      copts=tf_copts(),
      alwayslink=1,
      visibility=visibility)
  native.cc_library(
      name=name + "_internal",
      srcs=internalsrcs,
      hdrs=internalhdrs,
      deps=deps + if_not_android([
          "//tensorflow/core:core_cpu",
          "//tensorflow/core:framework",
          "//tensorflow/core:lib",
          "//tensorflow/core:protos_all_cc",
      ]) + if_android([
          "//tensorflow/core:android_tensorflow_lib",
      ]),
      copts=tf_copts(),
      alwayslink=1,
      visibility=["//tensorflow:internal"])


# Invoke this rule in .../tensorflow/python to build the wrapper library.
def tf_gen_op_wrapper_py(name,
                         out=None,
                         hidden=None,
                         visibility=None,
                         deps=[],
                         require_shape_functions=False,
                         hidden_file=None,
                         generated_target_name=None):
  # Construct a cc_binary containing the specified ops.
  tool_name = "gen_" + name + "_py_wrappers_cc"
  if not deps:
    deps = [str(Label("//tensorflow/core:" + name + "_op_lib"))]
  native.cc_binary(
      name=tool_name,
      linkopts=["-lm"],
      copts=tf_copts(),
      linkstatic=1,  # Faster to link this one-time-use binary dynamically
      deps=([
          "//tensorflow/core:framework",
          "//tensorflow/python:python_op_gen_main"
      ] + deps),
      visibility=["//tensorflow:internal"],)

  # Invoke the previous cc_binary to generate a python file.
  if not out:
    out = "ops/gen_" + name + ".py"

  if hidden:
    # `hidden` is a list of op names to be hidden in the generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        tools=[tool_name],
        cmd=("$(location " + tool_name + ") " + ",".join(hidden) + " " +
             ("1" if require_shape_functions else "0") + " > $@"))
  elif hidden_file:
    # `hidden_file` is file containing a list of op names to be hidden in the
    # generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        srcs=[hidden_file],
        tools=[tool_name],
        cmd=("$(location " + tool_name + ") @$(location " + hidden_file + ") " +
             ("1" if require_shape_functions else "0") + " > $@"))
  else:
    # No ops should be hidden in the generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        tools=[tool_name],
        cmd=("$(location " + tool_name + ") " +
             ("1" if require_shape_functions else "0") + " > $@"))

  # Make a py_library out of the generated python file.
  if not generated_target_name:
    generated_target_name = name
  native.py_library(
      name=generated_target_name,
      srcs=[out],
      srcs_version="PY2AND3",
      visibility=visibility,
      deps=[
          "//tensorflow/python:framework_for_generated_wrappers_v2",
      ],)


# Define a bazel macro that creates cc_test for tensorflow.
# TODO(opensource): we need to enable this to work around the hidden symbol
# __cudaRegisterFatBinary error. Need more investigations.
def tf_cc_test(name,
               srcs,
               deps,
               linkstatic=0,
               tags=[],
               data=[],
               size="medium",
               suffix="",
               args=None,
               linkopts=[]):
  native.cc_test(
      name="%s%s" % (name, suffix),
      srcs=srcs,
      size=size,
      args=args,
      copts=tf_copts(),
      data=data,
      deps=deps,
      linkopts=["-lpthread", "-lm"] + linkopts,
      linkstatic=linkstatic,
      tags=tags)


# Part of the testing workflow requires a distinguishable name for the build
# rules that involve a GPU, even if otherwise identical to the base rule.
def tf_cc_test_gpu(name,
                   srcs,
                   deps,
                   linkstatic=0,
                   tags=[],
                   data=[],
                   size="medium",
                   suffix="",
                   args=None):
  tf_cc_test(
      name,
      srcs,
      deps,
      linkstatic=linkstatic,
      tags=tags,
      data=data,
      size=size,
      suffix=suffix,
      args=args)


def tf_cuda_cc_test(name,
                    srcs=[],
                    deps=[],
                    tags=[],
                    data=[],
                    size="medium",
                    linkstatic=0,
                    args=[],
                    linkopts=[]):
  tf_cc_test(
      name=name,
      srcs=srcs,
      deps=deps,
      tags=tags + ["manual"],
      data=data,
      size=size,
      linkstatic=linkstatic,
      linkopts=linkopts,
      args=args)
  tf_cc_test(
      name=name,
      srcs=srcs,
      suffix="_gpu",
      deps=deps + if_cuda(["//tensorflow/core:gpu_runtime"]),
      linkstatic=if_cuda(1, 0),
      tags=tags + tf_cuda_tests_tags(),
      data=data,
      size=size,
      linkopts=linkopts,
      args=args)


# Create a cc_test for each of the tensorflow tests listed in "tests"
def tf_cc_tests(srcs,
                deps,
                name="",
                linkstatic=0,
                tags=[],
                size="medium",
                args=None,
                linkopts=[]):
  for src in srcs:
    tf_cc_test(
        name=src_to_test_name(src),
        srcs=[src],
        deps=deps,
        linkstatic=linkstatic,
        tags=tags,
        size=size,
        args=args,
        linkopts=linkopts)


def tf_cc_test_mkl(srcs,
                   deps,
                   name="",
                   linkstatic=0,
                   tags=[],
                   size="medium",
                   args=None):
  if_mkl(tf_cc_tests(srcs, deps, linkstatic, tags=tags, size=size, args=args))


def tf_cc_tests_gpu(srcs,
                    deps,
                    name="",
                    linkstatic=0,
                    tags=[],
                    size="medium",
                    args=None):
  tf_cc_tests(srcs, deps, linkstatic, tags=tags, size=size, args=args)


def tf_cuda_cc_tests(srcs,
                     deps,
                     name="",
                     tags=[],
                     size="medium",
                     linkstatic=0,
                     args=None,
                     linkopts=[]):
  for src in srcs:
    tf_cuda_cc_test(
        name=src_to_test_name(src),
        srcs=[src],
        deps=deps,
        tags=tags,
        size=size,
        linkstatic=linkstatic,
        args=args,
        linkopts=linkopts)


def _cuda_copts():
  """Gets the appropriate set of copts for (maybe) CUDA compilation.

    If we're doing CUDA compilation, returns copts for our particular CUDA
    compiler.  If we're not doing CUDA compilation, returns an empty list.

    """
  return cuda_default_copts() + select({
      "//conditions:default": [],
      "@local_config_cuda//cuda:using_nvcc": ([
          "-nvcc_options=relaxed-constexpr",
          "-nvcc_options=ftz=true",
      ]),
      "@local_config_cuda//cuda:using_clang": ([
          "-fcuda-flush-denormals-to-zero",
      ]),
  })


# Build defs for TensorFlow kernels


# When this target is built using --config=cuda, a cc_library is built
# that passes -DGOOGLE_CUDA=1 and '-x cuda', linking in additional
# libraries needed by GPU kernels.
def tf_gpu_kernel_library(srcs,
                          copts=[],
                          cuda_copts=[],
                          deps=[],
                          hdrs=[],
                          **kwargs):
  copts = copts + _cuda_copts() + if_cuda(cuda_copts) + tf_copts()

  native.cc_library(
      srcs=srcs,
      hdrs=hdrs,
      copts=copts,
      deps=deps + if_cuda([
          "//tensorflow/core:cuda",
          "//tensorflow/core:gpu_lib",
      ]),
      alwayslink=1,
      **kwargs)


def tf_cuda_library(deps=None, cuda_deps=None, copts=None, **kwargs):
  """Generate a cc_library with a conditional set of CUDA dependencies.

  When the library is built with --config=cuda:

  - both deps and cuda_deps are used as dependencies
  - the cuda runtime is added as a dependency (if necessary)
  - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts

  Args:
  - cuda_deps: BUILD dependencies which will be linked if and only if:
      '--config=cuda' is passed to the bazel command line.
  - deps: dependencies which will always be linked.
  - copts: copts always passed to the cc_library.
  - kwargs: Any other argument to cc_library.
  """
  if not deps:
    deps = []
  if not cuda_deps:
    cuda_deps = []
  if not copts:
    copts = []

  native.cc_library(
      deps=deps + if_cuda(cuda_deps + [
          "//tensorflow/core:cuda",
          "@local_config_cuda//cuda:cuda_headers"
      ]),
      copts=copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_mkl(["-DINTEL_MKL=1"]),
      **kwargs)


def tf_kernel_library(name,
                      prefix=None,
                      srcs=None,
                      gpu_srcs=None,
                      hdrs=None,
                      deps=None,
                      alwayslink=1,
                      copts=tf_copts(),
                      **kwargs):
  """A rule to build a TensorFlow OpKernel.

  May either specify srcs/hdrs or prefix.  Similar to tf_cuda_library,
  but with alwayslink=1 by default.  If prefix is specified:
    * prefix*.cc (except *.cu.cc) is added to srcs
    * prefix*.h (except *.cu.h) is added to hdrs
    * prefix*.cu.cc and prefix*.h (including *.cu.h) are added to gpu_srcs.
  With the exception that test files are excluded.
  For example, with prefix = "cast_op",
    * srcs = ["cast_op.cc"]
    * hdrs = ["cast_op.h"]
    * gpu_srcs = ["cast_op_gpu.cu.cc", "cast_op.h"]
    * "cast_op_test.cc" is excluded
  With prefix = "cwise_op"
    * srcs = ["cwise_op_abs.cc", ..., "cwise_op_tanh.cc"],
    * hdrs = ["cwise_ops.h", "cwise_ops_common.h"],
    * gpu_srcs = ["cwise_op_gpu_abs.cu.cc", ..., "cwise_op_gpu_tanh.cu.cc",
                  "cwise_ops.h", "cwise_ops_common.h",
                  "cwise_ops_gpu_common.cu.h"]
    * "cwise_ops_test.cc" is excluded
  """
  if not srcs:
    srcs = []
  if not hdrs:
    hdrs = []
  if not deps:
    deps = []

  if prefix:
    if native.glob([prefix + "*.cu.cc"], exclude=["*test*"]):
      if not gpu_srcs:
        gpu_srcs = []
      gpu_srcs = gpu_srcs + native.glob(
          [prefix + "*.cu.cc", prefix + "*.h"], exclude=[prefix + "*test*"])
    srcs = srcs + native.glob(
        [prefix + "*.cc"], exclude=[prefix + "*test*", prefix + "*.cu.cc"])
    hdrs = hdrs + native.glob(
        [prefix + "*.h"], exclude=[prefix + "*test*", prefix + "*.cu.h"])

  cuda_deps = [str(Label("//tensorflow/core:gpu_lib"))]
  if gpu_srcs:
    for gpu_src in gpu_srcs:
      if gpu_src.endswith(".cc") and not gpu_src.endswith(".cu.cc"):
        fail("{} not allowed in gpu_srcs. .cc sources must end with .cu.cc".
             format(gpu_src))
    tf_gpu_kernel_library(
        name=name + "_gpu", srcs=gpu_srcs, deps=deps, **kwargs)
    cuda_deps.extend([":" + name + "_gpu"])
  tf_cuda_library(
      name=name,
      srcs=srcs,
      hdrs=hdrs,
      copts=copts,
      cuda_deps=cuda_deps,
      linkstatic=1,  # Needed since alwayslink is broken in bazel b/27630669
      alwayslink=alwayslink,
      deps=deps,
      **kwargs)


def tf_mkl_kernel_library(name,
                          prefix=None,
                          srcs=None,
                          gpu_srcs=None,
                          hdrs=None,
                          deps=None,
                          alwayslink=1,
                          copts=tf_copts(),
                          **kwargs):
  if_mkl(
      tf_kernel_library(
          name,
          prefix=prefix,
          srcs=srcs,
          gpu_srcs=gpu_srcs,
          hdrs=hdrs,
          deps=deps,
          alwayslink=alwayslink,
          copts=copts,
          **kwargs))


# Bazel rules for building swig files.
def _py_wrap_cc_impl(ctx):
  srcs = ctx.files.srcs
  if len(srcs) != 1:
    fail("Exactly one SWIG source file label must be specified.", "srcs")
  module_name = ctx.attr.module_name
  src = ctx.files.srcs[0]
  inputs = set([src])
  inputs += ctx.files.swig_includes
  for dep in ctx.attr.deps:
    inputs += dep.cc.transitive_headers
  inputs += ctx.files._swiglib
  inputs += ctx.files.toolchain_deps
  swig_include_dirs = set(_get_repository_roots(ctx, inputs))
  swig_include_dirs += sorted([f.dirname for f in ctx.files._swiglib])
  args = [
      "-c++", "-python", "-module", module_name, "-o", ctx.outputs.cc_out.path,
      "-outdir", ctx.outputs.py_out.dirname
  ]
  args += ["-l" + f.path for f in ctx.files.swig_includes]
  args += ["-I" + i for i in swig_include_dirs]
  args += [src.path]
  outputs = [ctx.outputs.cc_out, ctx.outputs.py_out]
  ctx.action(
      executable=ctx.executable._swig,
      arguments=args,
      inputs=list(inputs),
      outputs=outputs,
      mnemonic="PythonSwig",
      progress_message="SWIGing " + src.path)
  return struct(files=set(outputs))


_py_wrap_cc = rule(
    attrs={
        "srcs":
            attr.label_list(
                mandatory=True,
                allow_files=True,),
        "swig_includes":
            attr.label_list(
                cfg="data",
                allow_files=True,),
        "deps":
            attr.label_list(
                allow_files=True,
                providers=["cc"],),
        "toolchain_deps":
            attr.label_list(
                allow_files=True,),
        "module_name":
            attr.string(mandatory=True),
        "py_module_name":
            attr.string(mandatory=True),
        "_swig":
            attr.label(
                default=Label("@swig//:swig"),
                executable=True,
                cfg="host",),
        "_swiglib":
            attr.label(
                default=Label("@swig//:templates"),
                allow_files=True,),
    },
    outputs={
        "cc_out": "%{module_name}.cc",
        "py_out": "%{py_module_name}.py",
    },
    implementation=_py_wrap_cc_impl,)


def _get_repository_roots(ctx, files):
  """Returns abnormal root directories under which files reside.

  When running a ctx.action, source files within the main repository are all
  relative to the current directory; however, files that are generated or exist
  in remote repositories will have their root directory be a subdirectory,
  e.g. bazel-out/local-fastbuild/genfiles/external/jpeg_archive. This function
  returns the set of these devious directories, ranked and sorted by popularity
  in order to hopefully minimize the number of I/O system calls within the
  compiler, because includes have quadratic complexity.
  """
  result = {}
  for f in files:
    root = f.root.path
    if root:
      if root not in result:
        result[root] = 0
      result[root] -= 1
    work = f.owner.workspace_root
    if work:
      if root:
        root += "/"
      root += work
    if root:
      if root not in result:
        result[root] = 0
      result[root] -= 1
  return [k for v, k in sorted([(v, k) for k, v in result.items()])]


# Bazel rule for collecting the header files that a target depends on.
def _transitive_hdrs_impl(ctx):
  outputs = set()
  for dep in ctx.attr.deps:
    outputs += dep.cc.transitive_headers
  return struct(files=outputs)


_transitive_hdrs = rule(
    attrs={
        "deps": attr.label_list(
            allow_files=True,
            providers=["cc"],),
    },
    implementation=_transitive_hdrs_impl,)


def transitive_hdrs(name, deps=[], **kwargs):
  _transitive_hdrs(name=name + "_gather", deps=deps)
  native.filegroup(name=name, srcs=[":" + name + "_gather"])


# Create a header only library that includes all the headers exported by
# the libraries in deps.
def cc_header_only_library(name, deps=[], **kwargs):
  _transitive_hdrs(name=name + "_gather", deps=deps)
  native.cc_library(name=name, hdrs=[":" + name + "_gather"], **kwargs)


def tf_custom_op_library_additional_deps():
  return [
      "@protobuf//:protobuf_headers",
      str(Label("//third_party/eigen3")),
      str(Label("//tensorflow/core:framework_headers_lib")),
  ]


# Traverse the dependency graph along the "deps" attribute of the
# target and return a struct with one field called 'tf_collected_deps'.
# tf_collected_deps will be the union of the deps of the current target
# and the tf_collected_deps of the dependencies of this target.
def _collect_deps_aspect_impl(target, ctx):
  alldeps = set()
  if hasattr(ctx.rule.attr, "deps"):
    for dep in ctx.rule.attr.deps:
      alldeps = alldeps | set([dep.label])
      if hasattr(dep, "tf_collected_deps"):
        alldeps = alldeps | dep.tf_collected_deps
  return struct(tf_collected_deps=alldeps)


collect_deps_aspect = aspect(
    implementation=_collect_deps_aspect_impl, attr_aspects=["deps"])


def _dep_label(dep):
  label = dep.label
  return label.package + ":" + label.name


# This rule checks that the transitive dependencies of targets listed
# in the 'deps' attribute don't depend on the targets listed in
# the 'disallowed_deps' attribute.
def _check_deps_impl(ctx):
  disallowed_deps = ctx.attr.disallowed_deps
  for input_dep in ctx.attr.deps:
    if not hasattr(input_dep, "tf_collected_deps"):
      continue
    for dep in input_dep.tf_collected_deps:
      for disallowed_dep in disallowed_deps:
        if dep == disallowed_dep.label:
          fail(
              _dep_label(input_dep) + " cannot depend on " + _dep_label(
                  disallowed_dep))
  return struct()


check_deps = rule(
    _check_deps_impl,
    attrs={
        "deps":
            attr.label_list(
                aspects=[collect_deps_aspect], mandatory=True,
                allow_files=True),
        "disallowed_deps":
            attr.label_list(mandatory=True, allow_files=True)
    },)


# Helper to build a dynamic library (.so) from the sources containing
# implementations of custom ops and kernels.
def tf_custom_op_library(name, srcs=[], gpu_srcs=[], deps=[]):
  cuda_deps = [
      str(Label("//tensorflow/core:stream_executor_headers_lib")),
      "@local_config_cuda//cuda:cudart_static",
  ]
  deps = deps + tf_custom_op_library_additional_deps()
  if gpu_srcs:
    basename = name.split(".")[0]
    native.cc_library(
        name=basename + "_gpu",
        srcs=gpu_srcs,
        copts=_cuda_copts(),
        deps=deps + if_cuda(cuda_deps))
    cuda_deps.extend([":" + basename + "_gpu"])

  check_deps(
      name=name + "_check_deps",
      deps=deps + if_cuda(cuda_deps),
      disallowed_deps=[
          "//tensorflow/core:framework",
          "//tensorflow/core:lib"
      ])

  native.cc_binary(
      name=name,
      srcs=srcs,
      deps=deps + if_cuda(cuda_deps),
      data=[name + "_check_deps"],
      copts=tf_copts(),
      linkshared=1,
      linkopts=select({
          "//conditions:default": [
              "-lm",
          ],
          "//tensorflow:darwin": [],
      }),)


def tf_custom_op_py_library(name,
                            srcs=[],
                            dso=[],
                            kernels=[],
                            srcs_version="PY2AND3",
                            visibility=None,
                            deps=[]):
  kernels = kernels  # unused argument
  native.py_library(
      name=name,
      data=dso,
      srcs=srcs,
      srcs_version=srcs_version,
      visibility=visibility,
      deps=deps,)


def tf_extension_linkopts():
  return []  # No extension link opts


def tf_extension_copts():
  return []  # No extension c opts


def tf_py_wrap_cc(name,
                             srcs,
                             swig_includes=[],
                             deps=[],
                             copts=[],
                             **kwargs):
  module_name = name.split("/")[-1]
  # Convert a rule name such as foo/bar/baz to foo/bar/_baz.so
  # and use that as the name for the rule producing the .so file.
  cc_library_name = "/".join(name.split("/")[:-1] + ["_" + module_name + ".so"])
  cc_library_pyd_name = "/".join(
      name.split("/")[:-1] + ["_" + module_name + ".pyd"])
  extra_deps = []
  _py_wrap_cc(
      name=name + "_py_wrap",
      srcs=srcs,
      swig_includes=swig_includes,
      deps=deps + extra_deps,
      toolchain_deps=["//tools/defaults:crosstool"],
      module_name=module_name,
      py_module_name=name)
  extra_linkopts = select({
      "@local_config_cuda//cuda:darwin": [
          "-Wl,-exported_symbols_list",
          str(Label("//tensorflow:tf_exported_symbols.lds"))
      ],
      str(Label("//tensorflow:windows")): [],
      "//conditions:default": [
          "-Wl,--version-script",
          "//tensorflow:tf_version_script.lds"
      ]
  })
  extra_deps += select({
      "@local_config_cuda//cuda:darwin": [
          "//tensorflow:tf_exported_symbols.lds"
      ],
      "//tensorflow:windows": [],
      "//conditions:default": [
          "//tensorflow:tf_version_script.lds"
      ]
  })

  native.cc_binary(
      name=cc_library_name,
      srcs=[module_name + ".cc"],
      copts=(copts + [
          "-Wno-self-assign", "-Wno-sign-compare", "-Wno-write-strings"
      ] + tf_extension_copts()),
      linkopts=tf_extension_linkopts() + extra_linkopts,
      linkstatic=1,
      linkshared=1,
      deps=deps + extra_deps)
  native.genrule(
      name="gen_" + cc_library_pyd_name,
      srcs=[":" + cc_library_name],
      outs=[cc_library_pyd_name],
      cmd="cp $< $@",)
  native.py_library(
      name=name,
      srcs=[":" + name + ".py"],
      srcs_version="PY2AND3",
      data=select({
          "//tensorflow:windows": [":" + cc_library_pyd_name],
          "//conditions:default": [":" + cc_library_name],
      }))


def py_test(deps=[], **kwargs):
  native.py_test(
      deps=select({
          "//conditions:default": deps,
          "//tensorflow:no_tensorflow_py_deps": []
      }),
      **kwargs)


def tf_py_test(name,
               srcs,
               size="medium",
               data=[],
               main=None,
               args=[],
               tags=[],
               shard_count=1,
               additional_deps=[],
               flaky=0,
               xla_enabled=False):
  if xla_enabled:
    additional_deps += tf_additional_xla_deps_py()
  native.py_test(
      name=name,
      size=size,
      srcs=srcs,
      main=main,
      args=args,
      tags=tags,
      visibility=[str(Label("//tensorflow:internal"))],
      shard_count=shard_count,
      data=data,
      deps=select({
          "//conditions:default": [
              "//tensorflow/python:extra_py_tests_deps",
              "//tensorflow/python:gradient_checker",
          ] + additional_deps,
          "//tensorflow:no_tensorflow_py_deps": []
      }),
      flaky=flaky,
      srcs_version="PY2AND3")


def cuda_py_test(name,
                 srcs,
                 size="medium",
                 data=[],
                 main=None,
                 args=[],
                 shard_count=1,
                 additional_deps=[],
                 tags=[],
                 flaky=0,
                 xla_enabled=False):
  test_tags = tags + tf_cuda_tests_tags()
  tf_py_test(
      name=name,
      size=size,
      srcs=srcs,
      data=data,
      main=main,
      args=args,
      tags=test_tags,
      shard_count=shard_count,
      additional_deps=additional_deps,
      flaky=flaky,
      xla_enabled=xla_enabled)


def sycl_py_test(name,
                 srcs,
                 size="medium",
                 data=[],
                 main=None,
                 args=[],
                 shard_count=1,
                 additional_deps=[],
                 tags=[],
                 flaky=0,
                 xla_enabled=False):
  test_tags = tags + tf_sycl_tests_tags()
  tf_py_test(
      name=name,
      size=size,
      srcs=srcs,
      data=data,
      main=main,
      args=args,
      tags=test_tags,
      shard_count=shard_count,
      additional_deps=additional_deps,
      flaky=flaky,
      xla_enabled=xla_enabled)


def py_tests(name,
             srcs,
             size="medium",
             additional_deps=[],
             data=[],
             tags=[],
             shard_count=1,
             prefix="",
             xla_enabled=False):
  for src in srcs:
    test_name = src.split("/")[-1].split(".")[0]
    if prefix:
      test_name = "%s_%s" % (prefix, test_name)
    tf_py_test(
        name=test_name,
        size=size,
        srcs=[src],
        main=src,
        tags=tags,
        shard_count=shard_count,
        data=data,
        additional_deps=additional_deps,
        xla_enabled=xla_enabled)


def cuda_py_tests(name,
                  srcs,
                  size="medium",
                  additional_deps=[],
                  data=[],
                  shard_count=1,
                  tags=[],
                  prefix="",
                  xla_enabled=False):
  test_tags = tags + tf_cuda_tests_tags()
  py_tests(
      name=name,
      size=size,
      srcs=srcs,
      additional_deps=additional_deps,
      data=data,
      tags=test_tags,
      shard_count=shard_count,
      prefix=prefix,
      xla_enabled=xla_enabled)


# Creates a genrule named <name> for running tools/proto_text's generator to
# make the proto_text functions, for the protos passed in <srcs>.
#
# Return a struct with fields (hdrs, srcs) containing the names of the
# generated files.
def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs):
  out_hdrs = (
      [p.replace(".proto", ".pb_text.h")
       for p in srcs] + [p.replace(".proto", ".pb_text-impl.h") for p in srcs])
  out_srcs = [p.replace(".proto", ".pb_text.cc") for p in srcs]
  native.genrule(
      name=name,
      srcs=srcs + ["//tensorflow/tools/proto_text:placeholder.txt"],
      outs=out_hdrs + out_srcs,
      cmd=
      "$(location //tensorflow/tools/proto_text:gen_proto_text_functions) "
      + "$(@D) " + srcs_relative_dir + " $(SRCS)",
      tools=[
          "//tensorflow/tools/proto_text:gen_proto_text_functions"
      ],)
  return struct(hdrs=out_hdrs, srcs=out_srcs)


def tf_genrule_cmd_append_to_srcs(to_append):
  return ("cat $(SRCS) > $(@) && " + "echo >> $(@) && " + "echo " + to_append +
          " >> $(@)")


def tf_version_info_genrule():
  native.genrule(
      name="version_info_gen",
      srcs=[
          "//tensorflow/tools/git:gen/spec.json",
          "//tensorflow/tools/git:gen/head",
          "//tensorflow/tools/git:gen/branch_ref",
      ],
      outs=["util/version_info.cc"],
      cmd=
      "$(location //tensorflow/tools/git:gen_git_source.py) --generate $(SRCS) \"$@\"",
      local=1,
      tools=["//tensorflow/tools/git:gen_git_source.py"],)


def cc_library_with_android_deps(deps,
                                 android_deps=[],
                                 common_deps=[],
                                 **kwargs):
  deps = if_not_android(deps) + if_android(android_deps) + common_deps
  native.cc_library(deps=deps, **kwargs)
