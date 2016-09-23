# -*- Python -*-

# Parse the bazel version string from `native.bazel_version`.
def _parse_bazel_version(bazel_version):
  # Remove commit from version.
  version = bazel_version.split(" ", 1)[0]

  # Split into (release, date) parts and only return the release
  # as a tuple of integers.
  parts = version.split('-', 1)

  # Turn "release" into a tuple of strings
  version_tuple = ()
  for number in parts[0].split('.'):
    version_tuple += (str(number),)
  return version_tuple

# Given a source file, generate a test name.
# i.e. "common_runtime/direct_session_test.cc" becomes
#      "common_runtime_direct_session_test"
def src_to_test_name(src):
  return src.replace("/", "_").split(".")[0]

# Check that a specific bazel version is being used.
def check_version(bazel_version):
  if "bazel_version" in dir(native) and native.bazel_version:
    current_bazel_version = _parse_bazel_version(native.bazel_version)
    minimum_bazel_version = _parse_bazel_version(bazel_version)
    if minimum_bazel_version > current_bazel_version:
      fail("\nCurrent Bazel version is {}, expected at least {}\n".format(
          native.bazel_version, bazel_version))
  pass

# Return the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

# List of proto files for android builds
def tf_android_core_proto_sources():
  return ["//tensorflow/core:" + p
          for p in tf_android_core_proto_sources_relative()]

# As tf_android_core_proto_sources, but paths relative to
# //third_party/tensorflow/core.
def tf_android_core_proto_sources_relative():
    return [
        "example/example.proto",
        "example/feature.proto",
        "framework/allocation_description.proto",
        "framework/attr_value.proto",
        "framework/cost_graph.proto",
        "framework/device_attributes.proto",
        "framework/function.proto",
        "framework/graph.proto",
        "framework/kernel_def.proto",
        "framework/log_memory.proto",
        "framework/node_def.proto",
        "framework/op_def.proto",
        "framework/step_stats.proto",
        "framework/summary.proto",
        "framework/tensor.proto",
        "framework/tensor_description.proto",
        "framework/tensor_shape.proto",
        "framework/tensor_slice.proto",
        "framework/types.proto",
        "framework/versions.proto",
        "lib/core/error_codes.proto",
        "protobuf/config.proto",
        "protobuf/saver.proto",
        "util/memmapped_file_system.proto",
        "util/saved_tensor_slice.proto",
  ]

# Returns the list of pb.h and proto.h headers that are generated for
# tf_android_core_proto_sources().
def tf_android_core_proto_headers():
  return (["//tensorflow/core/" + p.replace(".proto", ".pb.h")
          for p in tf_android_core_proto_sources_relative()] +
         ["//tensorflow/core/" + p.replace(".proto", ".proto.h")
          for p in tf_android_core_proto_sources_relative()])

# Returns the list of protos for which proto_text headers should be generated.
def tf_proto_text_protos_relative():
  return [p for p in tf_android_core_proto_sources_relative()]

def if_android_arm(a):
  return select({
      "//tensorflow:android_arm": a,
      "//conditions:default": [],
  })

def if_android_arm64(a):
  return select({
      "//tensorflow:android_arm64": a,
      "//conditions:default": [],
  })

def if_not_android(a):
  return select({
      "//tensorflow:android": [],
      "//conditions:default": a,
  })

def if_android(a):
  return select({
      "//tensorflow:android": a,
      "//conditions:default": [],
  })

def if_ios(a):
  return select({
      "//tensorflow:ios": a,
      "//conditions:default": [],
  })

def if_mobile(a):
  return select({
      "//tensorflow:android": a,
      "//tensorflow:ios": a,
      "//conditions:default": [],
  })

def if_not_mobile(a):
  return select({
      "//tensorflow:android": [],
      "//tensorflow:ios": [],
      "//conditions:default": a,
  })

def tf_copts():
  return (["-fno-exceptions", "-DEIGEN_AVOID_STL_ARRAY"] +
          if_cuda(["-DGOOGLE_CUDA=1"]) +
          if_android_arm(["-mfpu=neon"]) +
          select({"//tensorflow:android": [
                    "-std=c++11",
                    "-DMIN_LOG_LEVEL=0",
                    "-DTF_LEAN_BINARY",
                    "-O2",
                  ],
                  "//tensorflow:darwin": [],
                  "//tensorflow:ios": ["-std=c++11",],
                  "//conditions:default": ["-pthread"]}))

def tf_opts_nortti_if_android():
  return if_android([
      "-fno-rtti",
      "-DGOOGLE_PROTOBUF_NO_RTTI",
      "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
  ])

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(op_lib_names, deps=None):
  # Make library out of each op so it can also be used to generate wrappers
  # for various languages.
  if not deps:
    deps = []
  for n in op_lib_names:
    native.cc_library(name=n + "_op_lib",
                      copts=tf_copts(),
                      srcs=["ops/" + n + ".cc"],
                      deps=deps + ["//tensorflow/core:framework"],
                      visibility=["//visibility:public"],
                      alwayslink=1,
                      linkstatic=1,)

def tf_gen_op_wrapper_cc(name, out_ops_file, pkg="",
                         op_gen="//tensorflow/cc:cc_op_gen_main"):
  # Construct an op generator binary for these ops.
  tool = out_ops_file + "_gen_cc"
  native.cc_binary(
      name = tool,
      copts = tf_copts(),
      linkopts = ["-lm"],
      linkstatic = 1,   # Faster to link this one-time-use binary dynamically
      deps = ([op_gen, pkg + ":" + name + "_op_lib"])
  )

  # Run the op generator.
  if name == "sendrecv_ops":
    include_internal = "1"
  else:
    include_internal = "0"
  native.genrule(
      name=name + "_genrule",
      outs=[out_ops_file + ".h", out_ops_file + ".cc"],
      tools=[":" + tool],
      cmd=("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
           "$(location :" + out_ops_file + ".cc) " + include_internal))

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
def tf_gen_op_wrappers_cc(name,
                          op_lib_names=[],
                          other_srcs=[],
                          other_hdrs=[],
                          pkg="",
                          deps=[
                              "//tensorflow/cc:ops",
                              "//tensorflow/cc:scope",
                              "//tensorflow/cc:const_op",
                          ],
                          op_gen="//tensorflow/cc:cc_op_gen_main"):
  subsrcs = other_srcs
  subhdrs = other_hdrs
  for n in op_lib_names:
    tf_gen_op_wrapper_cc(n, "ops/" + n, pkg=pkg, op_gen=op_gen)
    subsrcs += ["ops/" + n + ".cc"]
    subhdrs += ["ops/" + n + ".h"]

  native.cc_library(name=name,
                    srcs=subsrcs,
                    hdrs=subhdrs,
                    deps=deps + [
                        "//tensorflow/core:core_cpu",
                        "//tensorflow/core:framework",
                        "//tensorflow/core:lib",
                        "//tensorflow/core:protos_all_cc",
                    ],
                    copts=tf_copts(),
                    alwayslink=1,)

# Invoke this rule in .../tensorflow/python to build the wrapper library.
def tf_gen_op_wrapper_py(name, out=None, hidden=None, visibility=None, deps=[],
                         require_shape_functions=False, hidden_file=None,
                         generated_target_name=None):
  # Construct a cc_binary containing the specified ops.
  tool_name = "gen_" + name + "_py_wrappers_cc"
  if not deps:
    deps = ["//tensorflow/core:" + name + "_op_lib"]
  native.cc_binary(
      name = tool_name,
      linkopts = ["-lm"],
      copts = tf_copts(),
      linkstatic = 1,   # Faster to link this one-time-use binary dynamically
      deps = (["//tensorflow/core:framework",
               "//tensorflow/python:python_op_gen_main"] + deps),
      visibility = ["//tensorflow:internal"],
  )

  # Invoke the previous cc_binary to generate a python file.
  if not out:
    out = "ops/gen_" + name + ".py"

  if hidden:
    # `hidden` is a list of op names to be hidden in the generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        tools=[tool_name],
        cmd=("$(location " + tool_name + ") " + ",".join(hidden)
             + " " + ("1" if require_shape_functions else "0") + " > $@"))
  elif hidden_file:
    # `hidden_file` is file containing a list of op names to be hidden in the
    # generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        srcs=[hidden_file],
        tools=[tool_name],
        cmd=("$(location " + tool_name + ") @$(location "
             + hidden_file + ") " + ("1" if require_shape_functions else "0")
             + " > $@"))
  else:
    # No ops should be hidden in the generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        tools=[tool_name],
        cmd=("$(location " + tool_name + ") "
             + ("1" if require_shape_functions else "0") + " > $@"))

  # Make a py_library out of the generated python file.
  if not generated_target_name:
    generated_target_name = name
  native.py_library(name=generated_target_name,
                    srcs=[out],
                    srcs_version="PY2AND3",
                    visibility=visibility,
                    deps=[
                        "//tensorflow/python:framework_for_generated_wrappers",
                    ],)

# Define a bazel macro that creates cc_test for tensorflow.
# TODO(opensource): we need to enable this to work around the hidden symbol
# __cudaRegisterFatBinary error. Need more investigations.
def tf_cc_test(name, srcs, deps, linkstatic=0, tags=[], data=[], size="medium",
               suffix="", args=None, linkopts=[]):
  native.cc_test(name="%s%s" % (name, suffix),
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
def tf_cc_test_gpu(name, srcs, deps, linkstatic=0, tags=[], data=[],
                   size="medium", suffix="", args=None):
  tf_cc_test(name, srcs, deps, linkstatic=linkstatic, tags=tags, data=data,
             size=size, suffix=suffix, args=args)

def tf_cuda_cc_test(name, srcs, deps, tags=[], data=[], size="medium",
                    linkstatic=0, args=[], linkopts=[]):
  tf_cc_test(name=name,
             srcs=srcs,
             deps=deps,
             tags=tags + ["manual"],
             data=data,
             size=size,
             linkstatic=linkstatic,
             linkopts=linkopts,
             args=args)
  tf_cc_test(name=name,
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
def tf_cc_tests(srcs, deps, linkstatic=0, tags=[], size="medium",
                args=None, linkopts=[]):
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

def tf_cc_tests_gpu(srcs, deps, linkstatic=0, tags=[], size="medium",
                    args=None):
  tf_cc_tests(srcs, deps, linkstatic, tags=tags, size=size, args=args)


def tf_cuda_cc_tests(srcs, deps, tags=[], size="medium", linkstatic=0,
                     args=None, linkopts=[]):
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
    common_cuda_opts = ["-x", "cuda", "-DGOOGLE_CUDA=1"]
    return select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": (
            common_cuda_opts +
            [
                "-nvcc_options=relaxed-constexpr",
                "-nvcc_options=ftz=true",
            ]
        ),
        "@local_config_cuda//cuda:using_clang": (
            common_cuda_opts +
            [
                "-fcuda-flush-denormals-to-zero",
                "--cuda-path=external/local_config_cuda/cuda",
                "--cuda-gpu-arch=sm_35",
            ]
        ),
    }) + select({
        # Pass -O3 when building CUDA code with clang; some important
        # optimizations are not enabled at O2.
        "@local_config_cuda//cuda:using_clang_opt": ["-O3"],
        "//conditions:default": [],
    })

# Build defs for TensorFlow kernels

# When this target is built using --config=cuda, a cc_library is built
# that passes -DGOOGLE_CUDA=1 and '-x cuda', linking in additional
# libraries needed by GPU kernels.
def tf_gpu_kernel_library(srcs, copts=[], cuda_copts=[], deps=[], hdrs=[],
                          **kwargs):
  copts = copts + _cuda_copts() + if_cuda(cuda_copts)

  native.cc_library(
      srcs = srcs,
      hdrs = hdrs,
      copts = copts,
      deps = deps + if_cuda([
          "//tensorflow/core:cuda",
          "//tensorflow/core:gpu_lib",
      ]),
      alwayslink=1,
      **kwargs)

def tf_cuda_library(deps=None, cuda_deps=None, copts=None, **kwargs):
  """Generate a cc_library with a conditional set of CUDA dependencies.

  When the library is built with --config=cuda:

  - both deps and cuda_deps are used as dependencies
  - the gcudacc runtime is added as a dependency (if necessary)
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
      deps = deps + if_cuda(cuda_deps + ["//tensorflow/core:cuda"]),
      copts = copts + if_cuda(["-DGOOGLE_CUDA=1"]),
      **kwargs)

def tf_kernel_library(name, prefix=None, srcs=None, gpu_srcs=None, hdrs=None,
                      deps=None, alwayslink=1, **kwargs):
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
    if native.glob([prefix + "*.cu.cc"], exclude = ["*test*"]):
      if not gpu_srcs:
        gpu_srcs = []
      gpu_srcs = gpu_srcs + native.glob([prefix + "*.cu.cc", prefix + "*.h"],
                                        exclude = ["*test*"])
    srcs = srcs + native.glob([prefix + "*.cc"],
                              exclude = ["*test*", "*.cu.cc"])
    hdrs = hdrs + native.glob([prefix + "*.h"], exclude = ["*test*", "*.cu.h"])

  cuda_deps = ["//tensorflow/core:gpu_lib"]
  if gpu_srcs:
    tf_gpu_kernel_library(
        name = name + "_gpu",
        srcs = gpu_srcs,
        deps = deps,
        **kwargs)
    cuda_deps.extend([":" + name + "_gpu"])
  tf_cuda_library(
      name = name,
      srcs = srcs,
      hdrs = hdrs,
      copts = tf_copts(),
      cuda_deps = cuda_deps,
      linkstatic = 1,   # Needed since alwayslink is broken in bazel b/27630669
      alwayslink = alwayslink,
      deps = deps,
      **kwargs)

def tf_kernel_libraries(name, prefixes, deps=None, **kwargs):
  """Makes one target per prefix, and one target that includes them all."""
  for p in prefixes:
    tf_kernel_library(name=p, prefix=p, deps=deps, **kwargs)
  native.cc_library(name=name, deps=[":" + p for p in prefixes])

# Bazel rules for building swig files.
def _py_wrap_cc_impl(ctx):
  srcs = ctx.files.srcs
  if len(srcs) != 1:
    fail("Exactly one SWIG source file label must be specified.", "srcs")
  module_name = ctx.attr.module_name
  cc_out = ctx.outputs.cc_out
  py_out = ctx.outputs.py_out
  src = ctx.files.srcs[0]
  args = ["-c++", "-python"]
  args += ["-module", module_name]
  args += ["-l" + f.path for f in ctx.files.swig_includes]
  cc_include_dirs = set()
  cc_includes = set()
  for dep in ctx.attr.deps:
    cc_include_dirs += [h.dirname for h in dep.cc.transitive_headers]
    cc_includes += dep.cc.transitive_headers
  args += ["-I" + x for x in cc_include_dirs]
  args += ["-I" + ctx.label.workspace_root]
  args += ["-o", cc_out.path]
  args += ["-outdir", py_out.dirname]
  args += [src.path]
  outputs = [cc_out, py_out]
  ctx.action(executable=ctx.executable.swig_binary,
             arguments=args,
             mnemonic="PythonSwig",
             inputs=sorted(set([src]) + cc_includes + ctx.files.swig_includes +
                         ctx.attr.swig_deps.files),
             outputs=outputs,
             progress_message="SWIGing {input}".format(input=src.path))
  return struct(files=set(outputs))

_py_wrap_cc = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "swig_includes": attr.label_list(
            cfg = "data",
            allow_files = True,
        ),
        "deps": attr.label_list(
            allow_files = True,
            providers = ["cc"],
        ),
        "swig_deps": attr.label(default = Label(
            "//tensorflow:swig",  # swig_templates
        )),
        "module_name": attr.string(mandatory = True),
        "py_module_name": attr.string(mandatory = True),
        "swig_binary": attr.label(
            default = Label("//tensorflow:swig"),
            cfg = "host",
            executable = True,
            allow_files = True,
        ),
    },
    outputs = {
        "cc_out": "%{module_name}.cc",
        "py_out": "%{py_module_name}.py",
    },
    implementation = _py_wrap_cc_impl,
)

# Bazel rule for collecting the header files that a target depends on.
def _transitive_hdrs_impl(ctx):
  outputs = set()
  for dep in ctx.attr.deps:
    outputs += dep.cc.transitive_headers
  return struct(files=outputs)

_transitive_hdrs = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = True,
            providers = ["cc"],
        ),
    },
    implementation = _transitive_hdrs_impl,
)

def transitive_hdrs(name, deps=[], **kwargs):
  _transitive_hdrs(name=name + "_gather",
                   deps=deps)
  native.filegroup(name=name,
                   srcs=[":" + name + "_gather"])

# Create a header only library that includes all the headers exported by
# the libraries in deps.
def cc_header_only_library(name, deps=[], **kwargs):
  _transitive_hdrs(name=name + "_gather",
                   deps=deps)
  native.cc_library(name=name,
                    hdrs=[":" + name + "_gather"],
                    **kwargs)

def tf_custom_op_library_additional_deps():
  return [
      "@protobuf//:protobuf",
      "//third_party/eigen3",
      "//tensorflow/core:framework_headers_lib",
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
    implementation=_collect_deps_aspect_impl,
    attr_aspects=["deps"])

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
          fail(_dep_label(input_dep) + " cannot depend on " +
               _dep_label(disallowed_dep))
  return struct()

check_deps = rule(
    _check_deps_impl,
    attrs = {
        "deps": attr.label_list(
            aspects=[collect_deps_aspect],
            mandatory = True,
            allow_files = True
        ),
        "disallowed_deps": attr.label_list(
            mandatory = True,
            allow_files = True
        )},
)

# Helper to build a dynamic library (.so) from the sources containing
# implementations of custom ops and kernels.
def tf_custom_op_library(name, srcs=[], gpu_srcs=[], deps=[]):
  cuda_deps = [
      "//tensorflow/core:stream_executor_headers_lib",
      "@local_config_cuda//cuda:cudart_static",
  ]
  deps = deps + tf_custom_op_library_additional_deps()
  if gpu_srcs:
    basename = name.split(".")[0]
    native.cc_library(
        name = basename + "_gpu",
        srcs = gpu_srcs,
        copts = _cuda_copts(),
        deps = deps + if_cuda(cuda_deps))
    cuda_deps.extend([":" + basename + "_gpu"])

  check_deps(name=name+"_check_deps",
             deps=deps + if_cuda(cuda_deps),
             disallowed_deps=["//tensorflow/core:framework",
                              "//tensorflow/core:lib"])

  native.cc_binary(name=name,
                   srcs=srcs,
                   deps=deps + if_cuda(cuda_deps),
                   data=[name + "_check_deps"],
                   copts=tf_copts(),
                   linkshared=1,
                   linkopts = select({
                       "//conditions:default": [
                           "-lm",
                       ],
                       "//tensorflow:darwin": [],
                   }),
  )

def tf_extension_linkopts():
  return []  # No extension link opts

def tf_extension_copts():
  return []  # No extension c opts

def tf_py_wrap_cc(name, srcs, swig_includes=[], deps=[], copts=[], **kwargs):
  module_name = name.split("/")[-1]
  # Convert a rule name such as foo/bar/baz to foo/bar/_baz.so
  # and use that as the name for the rule producing the .so file.
  cc_library_name = "/".join(name.split("/")[:-1] + ["_" + module_name + ".so"])
  extra_deps = []
  _py_wrap_cc(name=name + "_py_wrap",
              srcs=srcs,
              swig_includes=swig_includes,
              deps=deps + extra_deps,
              module_name=module_name,
              py_module_name=name)
  extra_linkopts = select({
      "@local_config_cuda//cuda:darwin": [
          "-Wl,-exported_symbols_list",
          "//tensorflow:tf_exported_symbols.lds"
      ],
      "//conditions:default": [
          "-Wl,--version-script",
          "//tensorflow:tf_version_script.lds"
      ]})
  extra_deps += select({
      "@local_config_cuda//cuda:darwin": [
        "//tensorflow:tf_exported_symbols.lds"
      ],
      "//conditions:default": [
        "//tensorflow:tf_version_script.lds"
      ]
  })

  native.cc_binary(
      name=cc_library_name,
      srcs=[module_name + ".cc"],
      copts=(copts + ["-Wno-self-assign", "-Wno-write-strings"]
             + tf_extension_copts()),
      linkopts=tf_extension_linkopts() + extra_linkopts,
      linkstatic=1,
      linkshared=1,
      deps=deps + extra_deps)
  native.py_library(name=name,
                    srcs=[":" + name + ".py"],
                    srcs_version="PY2AND3",
                    data=[":" + cc_library_name])

def tf_py_test(name, srcs, size="medium", data=[], main=None, args=[],
               tags=[], shard_count=1, additional_deps=[], flaky=0):
  native.py_test(
      name=name,
      size=size,
      srcs=srcs,
      main=main,
      args=args,
      tags=tags,
      visibility=["//tensorflow:internal"],
      shard_count=shard_count,
      data=data,
      deps=[
          "//tensorflow/python:extra_py_tests_deps",
          "//tensorflow/python:gradient_checker",
      ] + additional_deps,
      flaky=flaky,
      srcs_version="PY2AND3")

def cuda_py_test(name, srcs, size="medium", data=[], main=None, args=[],
                 shard_count=1, additional_deps=[], tags=[], flaky=0):
  test_tags = tags + tf_cuda_tests_tags()
  tf_py_test(name=name,
             size=size,
             srcs=srcs,
             data=data,
             main=main,
             args=args,
             tags=test_tags,
             shard_count=shard_count,
             additional_deps=additional_deps,
             flaky=flaky)

def py_tests(name,
             srcs,
             size="medium",
             additional_deps=[],
             data=[],
             tags=[],
             shard_count=1,
             prefix=""):
  for src in srcs:
    test_name = src.split("/")[-1].split(".")[0]
    if prefix:
      test_name = "%s_%s" % (prefix, test_name)
    tf_py_test(name=test_name,
               size=size,
               srcs=[src],
               main=src,
               tags=tags,
               shard_count=shard_count,
               data=data,
               additional_deps=additional_deps)

def cuda_py_tests(name, srcs, size="medium", additional_deps=[], data=[],
                  shard_count=1, tags=[], prefix=""):
  test_tags = tags + tf_cuda_tests_tags()
  py_tests(name=name, size=size, srcs=srcs, additional_deps=additional_deps,
           data=data, tags=test_tags, shard_count=shard_count,prefix=prefix)

# Creates a genrule named <name> for running tools/proto_text's generator to
# make the proto_text functions, for the protos passed in <srcs>.
#
# Return a struct with fields (hdrs, srcs) containing the names of the
# generated files.
def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs):
  out_hdrs = ([p.replace(".proto", ".pb_text.h") for p in srcs] +
              [p.replace(".proto", ".pb_text-impl.h") for p in srcs])
  out_srcs = [p.replace(".proto", ".pb_text.cc") for p in srcs]
  native.genrule(
        name = name,
        srcs = srcs + ["//tensorflow/tools/proto_text:placeholder.txt"],
        outs = out_hdrs + out_srcs,
        cmd = "$(location //tensorflow/tools/proto_text:gen_proto_text_functions) " +
              "$(@D) " + srcs_relative_dir + " $(SRCS)",
        tools = ["//tensorflow/tools/proto_text:gen_proto_text_functions"],
    )
  return struct(hdrs=out_hdrs, srcs=out_srcs)

def tf_genrule_cmd_append_to_srcs(to_append):
    return ("cat $(SRCS) > $(@) && " +
            "echo >> $(@) && " +
            "echo " + to_append + " >> $(@)")


def tf_version_info_genrule():
  native.genrule(
      name = "version_info_gen",
      srcs = [
          "//tensorflow/tools/git:gen/spec.json",
          "//tensorflow/tools/git:gen/head",
          "//tensorflow/tools/git:gen/branch_ref",
      ],
      outs = ["util/version_info.cc"],
      cmd = "$(location //tensorflow/tools/git:gen_git_source.py) --generate $(SRCS) \"$@\"",
      local = 1,
      tools = ["//tensorflow/tools/git:gen_git_source.py"],
  )
