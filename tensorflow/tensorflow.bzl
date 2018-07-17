# -*- Python -*-

# Return the options to use for a C++ library or binary build.
# Uses the ":optmode" config_setting to pick the options.
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_sycl_tests_tags",
    "tf_additional_grpc_deps_py",
    "tf_additional_xla_deps_py",
    "if_static",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
    "cuda_default_copts",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
    "if_mkl_lnx_x64"
)
load(
    "//third_party/mkl_dnn:build_defs.bzl",
    "if_mkl_open_source_only",
)
def register_extension_info(**kwargs):
    pass

# Given a source file, generate a test name.
# i.e. "common_runtime/direct_session_test.cc" becomes
#      "common_runtime_direct_session_test"
def src_to_test_name(src):
  return src.replace("/", "_").split(".")[0]

def full_path(relative_paths):
  return [native.package_name() + "/" + relative for relative in relative_paths]

def _add_tfcore_prefix(src):
  if src.startswith("//"):
    return src
  return "//tensorflow/core:" + src

# List of proto files for android builds
def tf_android_core_proto_sources(core_proto_sources_relative):
  return [
      _add_tfcore_prefix(p) for p in core_proto_sources_relative
  ]

# Returns the list of pb.h and proto.h headers that are generated for
# tf_android_core_proto_sources().
def tf_android_core_proto_headers(core_proto_sources_relative):
  return ([
      _add_tfcore_prefix(p).replace(":", "/").replace(".proto", ".pb.h")
      for p in core_proto_sources_relative
  ] + [
      _add_tfcore_prefix(p).replace(":", "/").replace(".proto", ".proto.h")
      for p in core_proto_sources_relative
  ])

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
  return str(Label(dep))

def if_android_x86(a):
  return select({
      clean_dep("//tensorflow:android_x86"): a,
      clean_dep("//tensorflow:android_x86_64"): a,
      "//conditions:default": [],
  })

def if_android_arm(a):
  return select({
      clean_dep("//tensorflow:android_arm"): a,
      "//conditions:default": [],
  })

def if_android_arm64(a):
  return select({
      clean_dep("//tensorflow:android_arm64"): a,
      "//conditions:default": [],
  })

def if_android_mips(a):
  return select({
      clean_dep("//tensorflow:android_mips"): a,
      "//conditions:default": [],
  })

def if_not_android(a):
  return select({
      clean_dep("//tensorflow:android"): [],
      "//conditions:default": a,
  })

def if_not_android_mips_and_mips64(a):
  return select({
      clean_dep("//tensorflow:android_mips"): [],
      clean_dep("//tensorflow:android_mips64"): [],
      "//conditions:default": a,
  })

def if_android(a):
  return select({
      clean_dep("//tensorflow:android"): a,
      "//conditions:default": [],
  })

def if_ios(a):
  return select({
      clean_dep("//tensorflow:ios"): a,
      "//conditions:default": [],
  })

def if_ios_x86_64(a):
  return select({
      clean_dep("//tensorflow:ios_x86_64"): a,
      "//conditions:default": [],
  })

def if_mobile(a):
  return select({
      clean_dep("//tensorflow:android"): a,
      clean_dep("//tensorflow:ios"): a,
      "//conditions:default": [],
  })

def if_not_mobile(a):
  return select({
      clean_dep("//tensorflow:android"): [],
      clean_dep("//tensorflow:ios"): [],
      "//conditions:default": a,
  })

def if_not_windows(a):
  return select({
      clean_dep("//tensorflow:windows"): [],
      clean_dep("//tensorflow:windows_msvc"): [],
      "//conditions:default": a,
  })

def if_windows(a):
  return select({
      clean_dep("//tensorflow:windows"): a,
      clean_dep("//tensorflow:windows_msvc"): a,
      "//conditions:default": [],
  })

def if_not_windows_cuda(a):
  return select({
      clean_dep("//tensorflow:with_cuda_support_windows_override"): [],
      "//conditions:default": a,
  })

def if_linux_x86_64(a):
  return select({
      clean_dep("//tensorflow:linux_x86_64"): a,
      "//conditions:default": [],
  })

def if_darwin(a):
  return select({
      clean_dep("//tensorflow:darwin"): a,
      "//conditions:default": [],
  })

def if_override_eigen_strong_inline(a):
  return select({
      clean_dep("//tensorflow:override_eigen_strong_inline"): a,
      "//conditions:default": [],
  })

def get_win_copts(is_external=False):
    WINDOWS_COPTS = [
        "/DPLATFORM_WINDOWS",
        "/DEIGEN_HAS_C99_MATH",
        "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
        "/DEIGEN_AVOID_STL_ARRAY",
        "/Iexternal/gemmlowp",
        "/wd4018",  # -Wno-sign-compare
        # Bazel's CROSSTOOL currently pass /EHsc to enable exception by
        # default. We can't pass /EHs-c- to disable exception, otherwise
        # we will get a waterfall of flag conflict warnings. Wait for
        # Bazel to fix this.
        # "/D_HAS_EXCEPTIONS=0",
        # "/EHs-c-",
        "/wd4577",
        "/DNOGDI",
    ]
    if is_external:
      return WINDOWS_COPTS + ["/UTF_COMPILE_LIBRARY"]
    else:
      return WINDOWS_COPTS + ["/DTF_COMPILE_LIBRARY"]

# LINT.IfChange
def tf_copts(android_optimization_level_override="-O2", is_external=False):
  # For compatibility reasons, android_optimization_level_override
  # is currently only being set for Android.
  # To clear this value, and allow the CROSSTOOL default
  # to be used, pass android_optimization_level_override=None
  android_copts = [
      "-std=c++11",
      "-DTF_LEAN_BINARY",
      "-Wno-narrowing",
      "-fomit-frame-pointer",
  ]
  if android_optimization_level_override:
    android_copts.append(android_optimization_level_override)
  return (
      if_not_windows([
          "-DEIGEN_AVOID_STL_ARRAY",
          "-Iexternal/gemmlowp",
          "-Wno-sign-compare",
          "-fno-exceptions",
          "-ftemplate-depth=900"])
      + if_cuda(["-DGOOGLE_CUDA=1"])
      + if_tensorrt(["-DGOOGLE_TENSORRT=1"])
      + if_mkl(["-DINTEL_MKL=1", "-DEIGEN_USE_VML"])
      + if_mkl_open_source_only(["-DDO_NOT_USE_ML"])
      + if_mkl_lnx_x64(["-fopenmp"])
      + if_android_arm(["-mfpu=neon"])
      + if_linux_x86_64(["-msse3"])
      + if_ios_x86_64(["-msse4.1"])
      + select({
            clean_dep("//tensorflow:framework_shared_object"): [],
            "//conditions:default": ["-DTENSORFLOW_MONOLITHIC_BUILD"],
      })
      + select({
            clean_dep("//tensorflow:android"): android_copts,
            clean_dep("//tensorflow:darwin"): [],
            clean_dep("//tensorflow:windows"): get_win_copts(is_external),
            clean_dep("//tensorflow:windows_msvc"): get_win_copts(is_external),
            clean_dep("//tensorflow:ios"): ["-std=c++11"],
            "//conditions:default": ["-pthread"]
      }))


def tfe_xla_copts():
  return select({
      "//tensorflow:with_xla_support": ["-DTENSORFLOW_EAGER_USE_XLA"],
      "//conditions:default": [],
  })

def tf_opts_nortti_if_android():
  return if_android([
      "-fno-rtti",
      "-DGOOGLE_PROTOBUF_NO_RTTI",
      "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
  ])

# LINT.ThenChange(//tensorflow/contrib/android/cmake/CMakeLists.txt)

def tf_features_nomodules_if_android():
  return if_android(["-use_header_modules"])

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(op_lib_names, deps=None, is_external=True):
  # Make library out of each op so it can also be used to generate wrappers
  # for various languages.
  if not deps:
    deps = []
  for n in op_lib_names:
    native.cc_library(
        name=n + "_op_lib",
        copts=tf_copts(is_external=is_external),
        srcs=["ops/" + n + ".cc"],
        deps=deps + [clean_dep("//tensorflow/core:framework")],
        visibility=["//visibility:public"],
        alwayslink=1,
        linkstatic=1,)

def _make_search_paths(prefix, levels_to_root):
  return ",".join(
      ["-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
       for search_level in range(levels_to_root + 1)])

def _rpath_linkopts(name):
  # Search parent directories up to the TensorFlow root directory for shared
  # object dependencies, even if this op shared object is deeply nested
  # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
  # the root and tensorflow/libtensorflow_framework.so should exist when
  # deployed. Other shared object dependencies (e.g. shared between contrib/
  # ops) are picked up as long as they are in either the same or a parent
  # directory in the tensorflow/ tree.
  levels_to_root = native.package_name().count("/") + name.count("/")
  return select({
      clean_dep("//tensorflow:darwin"): [
          "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
      ],
      clean_dep("//tensorflow:windows"): [],
      clean_dep("//tensorflow:windows_msvc"): [],
      "//conditions:default": [
          "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),
      ],
  })

# Bazel-generated shared objects which must be linked into TensorFlow binaries
# to define symbols from //tensorflow/core:framework and //tensorflow/core:lib.
def tf_binary_additional_srcs():
  return if_static(
      extra_deps=[],
      otherwise=[
          clean_dep("//tensorflow:libtensorflow_framework.so"),
      ])

def tf_cc_shared_object(
    name,
    srcs=[],
    deps=[],
    linkopts=[],
    framework_so=tf_binary_additional_srcs(),
    **kwargs):
  native.cc_binary(
      name=name,
      srcs=srcs + framework_so,
      deps=deps,
      linkshared = 1,
      linkopts=linkopts + _rpath_linkopts(name) + select({
          clean_dep("//tensorflow:darwin"): [
              "-Wl,-install_name,@rpath/" + name.split("/")[-1],
          ],
          clean_dep("//tensorflow:windows"): [],
          "//conditions:default": [
              "-Wl,-soname," + name.split("/")[-1],
          ],
      }),
      **kwargs)

register_extension_info(
    extension_name = "tf_cc_shared_object",
    label_regex_for_dep = "{extension_name}",
)

# Links in the framework shared object
# (//third_party/tensorflow:libtensorflow_framework.so) when not building
# statically. Also adds linker options (rpaths) so that the framework shared
# object can be found.
def tf_cc_binary(name,
                 srcs=[],
                 deps=[],
                 linkopts=[],
                 copts=tf_copts(),
                 **kwargs):
  native.cc_binary(
      name=name,
      copts=copts,
      srcs=srcs + tf_binary_additional_srcs(),
      deps=deps + if_mkl(
          [
              "//third_party/mkl:intel_binary_blob",
          ],
      ),
      linkopts=linkopts + _rpath_linkopts(name),
      **kwargs)

register_extension_info(
    extension_name = "tf_cc_binary",
    label_regex_for_dep = "{extension_name}.*",
)

# A simple wrap around native.cc_binary rule.
# When using this rule, you should realize it doesn't link to any tensorflow
# dependencies by default.
def tf_native_cc_binary(name,
                        copts=tf_copts(),
                        **kwargs):
  native.cc_binary(
      name=name,
      copts=copts,
      **kwargs)

register_extension_info(
    extension_name = "tf_native_cc_binary",
    label_regex_for_dep = "{extension_name}.*",
)

def tf_gen_op_wrapper_cc(name,
                         out_ops_file,
                         pkg="",
                         op_gen=clean_dep("//tensorflow/cc:cc_op_gen_main"),
                         deps=None,
                         include_internal_ops=0,
                         # ApiDefs will be loaded in the order specified in this list.
                         api_def_srcs=[]):
  # Construct an op generator binary for these ops.
  tool = out_ops_file + "_gen_cc"
  if deps == None:
    deps = [pkg + ":" + name + "_op_lib"]
  tf_cc_binary(
      name=tool,
      copts=tf_copts(),
      linkopts=if_not_windows(["-lm"]),
      linkstatic=1,  # Faster to link this one-time-use binary dynamically
      deps=[op_gen] + deps)

  srcs = api_def_srcs[:]

  if not api_def_srcs:
    api_def_args_str = ","
  else:
    api_def_args = []
    for api_def_src in api_def_srcs:
      # Add directory of the first ApiDef source to args.
      # We are assuming all ApiDefs in a single api_def_src are in the
      # same directory.
      api_def_args.append(
          " $$(dirname $$(echo $(locations " + api_def_src +
          ") | cut -d\" \" -f1))")
    api_def_args_str = ",".join(api_def_args)

  native.genrule(
      name=name + "_genrule",
      outs=[
          out_ops_file + ".h", out_ops_file + ".cc",
          out_ops_file + "_internal.h", out_ops_file + "_internal.cc"
      ],
      srcs=srcs,
      tools=[":" + tool] + tf_binary_additional_srcs(),
      cmd=("$(location :" + tool + ") $(location :" + out_ops_file + ".h) " +
           "$(location :" + out_ops_file + ".cc) " +
           str(include_internal_ops) + " " + api_def_args_str))

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
# TODO(joshl): Cleaner approach for hidden ops.
def tf_gen_op_wrappers_cc(name,
                          op_lib_names=[],
                          other_srcs=[],
                          other_hdrs=[],
                          pkg="",
                          deps=[
                              clean_dep("//tensorflow/cc:ops"),
                              clean_dep("//tensorflow/cc:scope"),
                              clean_dep("//tensorflow/cc:const_op"),
                          ],
                          op_gen=clean_dep("//tensorflow/cc:cc_op_gen_main"),
                          include_internal_ops=0,
                          visibility=None,
                          # ApiDefs will be loaded in the order apecified in this list.
                          api_def_srcs=[]):
  subsrcs = other_srcs[:]
  subhdrs = other_hdrs[:]
  internalsrcs = []
  internalhdrs = []
  for n in op_lib_names:
    tf_gen_op_wrapper_cc(
        n,
        "ops/" + n,
        pkg=pkg,
        op_gen=op_gen,
        include_internal_ops=include_internal_ops,
        api_def_srcs=api_def_srcs)
    subsrcs += ["ops/" + n + ".cc"]
    subhdrs += ["ops/" + n + ".h"]
    internalsrcs += ["ops/" + n + "_internal.cc"]
    internalhdrs += ["ops/" + n + "_internal.h"]

  native.cc_library(
      name=name,
      srcs=subsrcs,
      hdrs=subhdrs,
      deps=deps + if_not_android([
          clean_dep("//tensorflow/core:core_cpu"),
          clean_dep("//tensorflow/core:framework"),
          clean_dep("//tensorflow/core:lib"),
          clean_dep("//tensorflow/core:protos_all_cc"),
      ]) + if_android([
          clean_dep("//tensorflow/core:android_tensorflow_lib"),
      ]),
      copts=tf_copts(),
      alwayslink=1,
      visibility=visibility)
  native.cc_library(
      name=name + "_internal",
      srcs=internalsrcs,
      hdrs=internalhdrs,
      deps=deps + if_not_android([
          clean_dep("//tensorflow/core:core_cpu"),
          clean_dep("//tensorflow/core:framework"),
          clean_dep("//tensorflow/core:lib"),
          clean_dep("//tensorflow/core:protos_all_cc"),
      ]) + if_android([
          clean_dep("//tensorflow/core:android_tensorflow_lib"),
      ]),
      copts=tf_copts(),
      alwayslink=1,
      visibility=[clean_dep("//tensorflow:internal")])

# Generates a Python library target wrapping the ops registered in "deps".
#
# Args:
#   name: used as the name of the generated target and as a name component of
#     the intermediate files.
#   out: name of the python file created by this rule. If None, then
#     "ops/gen_{name}.py" is used.
#   hidden: Optional list of ops names to make private in the Python module.
#     It is invalid to specify both "hidden" and "op_whitelist".
#   visibility: passed to py_library.
#   deps: list of dependencies for the intermediate tool used to generate the
#     python target. NOTE these `deps` are not applied to the final python
#     library target itself.
#   require_shape_functions: leave this as False.
#   hidden_file: optional file that contains a list of op names to make private
#     in the generated Python module. Each op name should be on a line by
#     itself. Lines that start with characters that are invalid op name
#     starting characters are treated as comments and ignored.
#   generated_target_name: name of the generated target (overrides the
#     "name" arg)
#   op_whitelist: if not empty, only op names in this list will be wrapped. It
#     is invalid to specify both "hidden" and "op_whitelist".
#   cc_linkopts: Optional linkopts to be added to tf_cc_binary that contains the
#     specified ops.
#   gen_locally: if True, the genrule to generate the Python library will be run
#     without sandboxing. This would help when the genrule depends on symlinks
#     which may not be supported in the sandbox.
def tf_gen_op_wrapper_py(name,
                         out=None,
                         hidden=None,
                         visibility=None,
                         deps=[],
                         require_shape_functions=False,
                         hidden_file=None,
                         generated_target_name=None,
                         op_whitelist=[],
                         cc_linkopts=[],
                         api_def_srcs=[],
                         gen_locally=False):
  if (hidden or hidden_file) and op_whitelist:
    fail('Cannot pass specify both hidden and op_whitelist.')

  # Construct a cc_binary containing the specified ops.
  tool_name = "gen_" + name + "_py_wrappers_cc"
  if not deps:
    deps = [str(Label("//tensorflow/core:" + name + "_op_lib"))]
  tf_cc_binary(
      name=tool_name,
      linkopts=if_not_windows(["-lm"]) + cc_linkopts,
      copts=tf_copts(),
      linkstatic=1,  # Faster to link this one-time-use binary dynamically
      deps=([
          clean_dep("//tensorflow/core:framework"),
          clean_dep("//tensorflow/python:python_op_gen_main")
      ] + deps),
      visibility=[clean_dep("//tensorflow:internal")],)

  # Invoke the previous cc_binary to generate a python file.
  if not out:
    out = "ops/gen_" + name + ".py"

  if hidden:
    op_list_arg = ",".join(hidden)
    op_list_is_whitelist = False
  elif op_whitelist:
    op_list_arg = ",".join(op_whitelist)
    op_list_is_whitelist = True
  else:
    op_list_arg = "''"
    op_list_is_whitelist = False

  # Prepare ApiDef directories to pass to the genrule.
  if not api_def_srcs:
    api_def_args_str = ","
  else:
    api_def_args = []
    for api_def_src in api_def_srcs:
      # Add directory of the first ApiDef source to args.
      # We are assuming all ApiDefs in a single api_def_src are in the
      # same directory.
      api_def_args.append(
          "$$(dirname $$(echo $(locations " + api_def_src +
          ") | cut -d\" \" -f1))")
    api_def_args_str = ",".join(api_def_args)

  if hidden_file:
    # `hidden_file` is file containing a list of op names to be hidden in the
    # generated module.
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        srcs=api_def_srcs + [hidden_file],
        tools=[tool_name] + tf_binary_additional_srcs(),
        local = (1 if gen_locally else 0),
        cmd=("$(location " + tool_name + ") " + api_def_args_str +
             " @$(location " + hidden_file + ") " +
             ("1" if require_shape_functions else "0") + " > $@"))
  else:
    native.genrule(
        name=name + "_pygenrule",
        outs=[out],
        srcs=api_def_srcs,
        tools=[tool_name] + tf_binary_additional_srcs(),
        local = (1 if gen_locally else 0),
        cmd=("$(location " + tool_name + ") " + api_def_args_str + " " +
             op_list_arg + " " +
             ("1" if require_shape_functions else "0") + " " +
             ("1" if op_list_is_whitelist else "0") + " > $@"))

  # Make a py_library out of the generated python file.
  if not generated_target_name:
    generated_target_name = name
  native.py_library(
      name=generated_target_name,
      srcs=[out],
      srcs_version="PY2AND3",
      visibility=visibility,
      deps=[
          clean_dep("//tensorflow/python:framework_for_generated_wrappers_v2"),
      ],)

# Define a bazel macro that creates cc_test for tensorflow.
#
# Links in the framework shared object
# (//third_party/tensorflow:libtensorflow_framework.so) when not building
# statically. Also adds linker options (rpaths) so that the framework shared
# object can be found.
#
# TODO(opensource): we need to enable this to work around the hidden symbol
# __cudaRegisterFatBinary error. Need more investigations.
def tf_cc_test(name,
               srcs,
               deps,
               linkstatic=0,
               extra_copts=[],
               suffix="",
               linkopts=[],
               nocopts=None,
               **kwargs):
  native.cc_test(
      name="%s%s" % (name, suffix),
      srcs=srcs + tf_binary_additional_srcs(),
      copts=tf_copts() + extra_copts,
      linkopts=select({
        clean_dep("//tensorflow:android"): [
            "-pie",
        ],
        clean_dep("//tensorflow:windows"): [],
        clean_dep("//tensorflow:windows_msvc"): [],
        clean_dep("//tensorflow:darwin"): [
            "-lm",
        ],
        "//conditions:default": [
            "-lpthread",
            "-lm"
        ],
      }) + linkopts + _rpath_linkopts(name),
      deps=deps + if_mkl(
          [
              "//third_party/mkl:intel_binary_blob",
          ],
      ),
      # Nested select() statements seem not to be supported when passed to
      # linkstatic, and we already have a cuda select() passed in to this
      # function.
      linkstatic=linkstatic or select({
          # cc_tests with ".so"s in srcs incorrectly link on Darwin unless
          # linkstatic=1 (https://github.com/bazelbuild/bazel/issues/3450).
          # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
          clean_dep("//tensorflow:darwin"): 1,
          "//conditions:default": 0,
      }),
      nocopts=nocopts,
      **kwargs)

register_extension_info(
    extension_name = "tf_cc_test",
    label_regex_for_dep = "{extension_name}.*",
)

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

register_extension_info(
    extension_name = "tf_cc_test_gpu",
    label_regex_for_dep = "{extension_name}",
)

def tf_cuda_cc_test(name,
                    srcs=[],
                    deps=[],
                    tags=[],
                    data=[],
                    size="medium",
                    extra_copts=[],
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
      extra_copts=extra_copts,
      linkstatic=linkstatic,
      linkopts=linkopts,
      args=args)
  tf_cc_test(
      name=name,
      srcs=srcs,
      suffix="_gpu",
      deps=deps + if_cuda([
          clean_dep("//tensorflow/core:gpu_runtime"),
      ]),
      linkstatic=select({
          # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
          clean_dep("//tensorflow:darwin"): 1,
          "@local_config_cuda//cuda:using_nvcc": 1,
          "@local_config_cuda//cuda:using_clang": 1,
          "//conditions:default": 0,
      }),
      tags=tags + tf_cuda_tests_tags(),
      data=data,
      size=size,
      extra_copts=extra_copts,
      linkopts=linkopts,
      args=args)

register_extension_info(
    extension_name = "tf_cuda_cc_test",
    label_regex_for_dep = "{extension_name}",
)

def tf_cuda_only_cc_test(name,
                    srcs=[],
                    deps=[],
                    tags=[],
                    data=[],
                    size="medium",
                    linkstatic=0,
                    args=[],
                    linkopts=[]):
  native.cc_test(
      name="%s%s" % (name, "_gpu"),
      srcs=srcs + tf_binary_additional_srcs(),
      size=size,
      args=args,
      copts= _cuda_copts() + tf_copts(),
      data=data,
      deps=deps + if_cuda([
          clean_dep("//tensorflow/core:cuda"),
          clean_dep("//tensorflow/core:gpu_lib")]),
      linkopts=if_not_windows(["-lpthread", "-lm"]) + linkopts + _rpath_linkopts(name),
      linkstatic=linkstatic or select({
          # cc_tests with ".so"s in srcs incorrectly link on Darwin
          # unless linkstatic=1.
          # TODO(allenl): Remove Mac static linking when Bazel 0.6 is out.
          clean_dep("//tensorflow:darwin"): 1,
          "//conditions:default": 0,
      }),
      tags=tags + tf_cuda_tests_tags())

register_extension_info(
    extension_name = "tf_cuda_only_cc_test",
    label_regex_for_dep = "{extension_name}_gpu",
)

# Create a cc_test for each of the tensorflow tests listed in "tests"
def tf_cc_tests(srcs,
                deps,
                name="",
                linkstatic=0,
                tags=[],
                size="medium",
                args=None,
                linkopts=[],
                nocopts=None):
  for src in srcs:
    tf_cc_test(
        name=src_to_test_name(src),
        srcs=[src],
        deps=deps,
        linkstatic=linkstatic,
        tags=tags,
        size=size,
        args=args,
        linkopts=linkopts,
        nocopts=nocopts)

def tf_cc_test_mkl(srcs,
                   deps,
                   name="",
                   linkstatic=0,
                   tags=[],
                   size="medium",
                   args=None):
  # -fno-exceptions in nocopts breaks compilation if header modules are enabled.
  disable_header_modules = ["-use_header_modules"]

  for src in srcs:
    native.cc_test(
      name=src_to_test_name(src),
      srcs=if_mkl([src]) + tf_binary_additional_srcs(),
      copts=tf_copts(),
      linkopts=select({
        clean_dep("//tensorflow:android"): [
            "-pie",
          ],
        clean_dep("//tensorflow:windows"): [],
        clean_dep("//tensorflow:windows_msvc"): [],
        "//conditions:default": [
            "-lpthread",
            "-lm"
        ],
      }) + _rpath_linkopts(src_to_test_name(src)),
      deps=deps + if_mkl(
          [
              "//third_party/mkl:intel_binary_blob",
          ],
      ),
      linkstatic=linkstatic,
      tags=tags,
      size=size,
      args=args,
      features=disable_header_modules,
      nocopts="-fno-exceptions")


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

def tf_java_test(name,
                 srcs=[],
                 deps=[],
                 *args,
                 **kwargs):
  native.java_test(
      name=name,
      srcs=srcs,
      deps=deps + tf_binary_additional_srcs(),
      *args,
      **kwargs)

register_extension_info(
    extension_name = "tf_java_test",
    label_regex_for_dep = "{extension_name}",
)

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
  kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]

  native.cc_library(
      srcs=srcs,
      hdrs=hdrs,
      copts=copts,
      deps=deps + if_cuda([
          clean_dep("//tensorflow/core:cuda"),
          clean_dep("//tensorflow/core:gpu_lib"),
      ]),
      alwayslink=1,
      **kwargs)

register_extension_info(
    extension_name = "tf_gpu_kernel_library",
    label_regex_for_dep = "{extension_name}",
)

def tf_cuda_library(deps=None, cuda_deps=None, copts=tf_copts(), **kwargs):
  """Generate a cc_library with a conditional set of CUDA dependencies.

  When the library is built with --config=cuda:

  - Both deps and cuda_deps are used as dependencies.
  - The cuda runtime is added as a dependency (if necessary).
  - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts.
  - In addition, when the library is also built with TensorRT enabled, it
      additionally passes -DGOOGLE_TENSORRT=1 to the list of copts.

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

  kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
  native.cc_library(
      deps=deps + if_cuda(cuda_deps + [
          clean_dep("//tensorflow/core:cuda"),
          "@local_config_cuda//cuda:cuda_headers"
      ]),
      copts=(copts + if_cuda(["-DGOOGLE_CUDA=1"]) + if_mkl(["-DINTEL_MKL=1"]) +
             if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
      **kwargs)

register_extension_info(
    extension_name = "tf_cuda_library",
    label_regex_for_dep = "{extension_name}",
)

def tf_kernel_library(
        name,
        prefix = None,
        srcs = None,
        gpu_srcs = None,
        hdrs = None,
        deps = None,
        alwayslink = 1,
        copts = None,
        is_external = False,
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
  if not copts:
    copts = []
  textual_hdrs = []
  copts = copts + tf_copts(is_external=is_external)
  if prefix:
    if native.glob([prefix + "*.cu.cc"], exclude=["*test*"]):
      if not gpu_srcs:
        gpu_srcs = []
      gpu_srcs = gpu_srcs + native.glob(
          [prefix + "*.cu.cc", prefix + "*.h"], exclude=[prefix + "*test*"])
    srcs = srcs + native.glob(
        [prefix + "*.cc"], exclude=[prefix + "*test*", prefix + "*.cu.cc"])
    hdrs = hdrs + native.glob(
            [prefix + "*.h"],
            exclude = [prefix + "*test*", prefix + "*.cu.h", prefix + "*impl.h"],
        )
    textual_hdrs = native.glob(
            [prefix + "*impl.h"],
            exclude = [prefix + "*test*", prefix + "*.cu.h"],
        )
  cuda_deps = [clean_dep("//tensorflow/core:gpu_lib")]
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
      textual_hdrs = textual_hdrs,
      copts=copts,
      cuda_deps=cuda_deps,
      linkstatic=1,  # Needed since alwayslink is broken in bazel b/27630669
      alwayslink=alwayslink,
      deps=deps,
      **kwargs)

register_extension_info(
    extension_name = "tf_kernel_library",
    label_regex_for_dep = "{extension_name}(_gpu)?",
)

def tf_mkl_kernel_library(name,
                          prefix=None,
                          srcs=None,
                          hdrs=None,
                          deps=None,
                          alwayslink=1,
                          copts=tf_copts(),
                          nocopts="-fno-exceptions"):
  """A rule to build MKL-based TensorFlow kernel libraries."""

  if not bool(srcs):
    srcs = []
  if not bool(hdrs):
    hdrs = []

  if prefix:
    srcs = srcs + native.glob(
        [prefix + "*.cc"])
    hdrs = hdrs + native.glob(
        [prefix + "*.h"])

  # -fno-exceptions in nocopts breaks compilation if header modules are enabled.
  disable_header_modules = ["-use_header_modules"]

  native.cc_library(
      name=name,
      srcs=if_mkl(srcs),
      hdrs=hdrs,
      deps=deps,
      alwayslink=alwayslink,
      copts=copts,
      nocopts=nocopts,
      features = disable_header_modules
  )

register_extension_info(
    extension_name = "tf_mkl_kernel_library",
    label_regex_for_dep = "{extension_name}",
)

# Bazel rules for building swig files.
def _py_wrap_cc_impl(ctx):
  srcs = ctx.files.srcs
  if len(srcs) != 1:
    fail("Exactly one SWIG source file label must be specified.", "srcs")
  module_name = ctx.attr.module_name
  src = ctx.files.srcs[0]
  inputs = depset([src])
  inputs += ctx.files.swig_includes
  for dep in ctx.attr.deps:
    inputs += dep.cc.transitive_headers
  inputs += ctx.files._swiglib
  inputs += ctx.files.toolchain_deps
  swig_include_dirs = depset(_get_repository_roots(ctx, inputs))
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
  return struct(files=depset(outputs))

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
        "toolchain_deps": attr.label_list(
            allow_files = True,
        ),
        "module_name": attr.string(mandatory = True),
        "py_module_name": attr.string(mandatory = True),
        "_swig": attr.label(
            default = Label("@swig//:swig"),
            executable = True,
            cfg = "host",
        ),
        "_swiglib": attr.label(
            default = Label("@swig//:templates"),
            allow_files = True,
        ),
    },
    outputs = {
        "cc_out": "%{module_name}.cc",
        "py_out": "%{py_module_name}.py",
    },
    implementation = _py_wrap_cc_impl,
)

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
  outputs = depset()
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
  _transitive_hdrs(name=name + "_gather", deps=deps)
  native.filegroup(name=name, srcs=[":" + name + "_gather"])

# Create a header only library that includes all the headers exported by
# the libraries in deps.
def cc_header_only_library(name, deps=[], includes=[], **kwargs):
  _transitive_hdrs(name=name + "_gather", deps=deps)
  native.cc_library(name=name,
                    hdrs=[":" + name + "_gather"],
                    includes=includes,
                    **kwargs)

def tf_custom_op_library_additional_deps():
  return [
      "@protobuf_archive//:protobuf_headers",
      clean_dep("//third_party/eigen3"),
      clean_dep("//tensorflow/core:framework_headers_lib"),
  ] + if_windows(["//tensorflow/python:pywrap_tensorflow_import_lib"])

# A list of targets that contains the implemenation of
# tf_custom_op_library_additional_deps. It's used to generate a DEF file for
# exporting symbols from _pywrap_tensorflow.dll on Windows.
def tf_custom_op_library_additional_deps_impl():
  return [
      "@protobuf_archive//:protobuf",
      "@nsync//:nsync_cpp",
      # for //third_party/eigen3
      clean_dep("//third_party/eigen3"),
      # for //tensorflow/core:framework_headers_lib
      clean_dep("//tensorflow/core:framework"),
      clean_dep("//tensorflow/core:reader_base"),
  ]

# Traverse the dependency graph along the "deps" attribute of the
# target and return a struct with one field called 'tf_collected_deps'.
# tf_collected_deps will be the union of the deps of the current target
# and the tf_collected_deps of the dependencies of this target.
def _collect_deps_aspect_impl(target, ctx):
  alldeps = depset()
  if hasattr(ctx.rule.attr, "deps"):
    for dep in ctx.rule.attr.deps:
      alldeps = alldeps | depset([dep.label])
      if hasattr(dep, "tf_collected_deps"):
        alldeps = alldeps | dep.tf_collected_deps
  return struct(tf_collected_deps=alldeps)

collect_deps_aspect = aspect(
    attr_aspects = ["deps"],
    implementation = _collect_deps_aspect_impl,
)

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
    attrs = {
        "deps": attr.label_list(
            aspects = [collect_deps_aspect],
            mandatory = True,
            allow_files = True,
        ),
        "disallowed_deps": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
    },
)

# Helper to build a dynamic library (.so) from the sources containing
# implementations of custom ops and kernels.
def tf_custom_op_library(name, srcs=[], gpu_srcs=[], deps=[], linkopts=[]):
  cuda_deps = [
      clean_dep("//tensorflow/core:stream_executor_headers_lib"),
      "@local_config_cuda//cuda:cuda_headers",
      "@local_config_cuda//cuda:cudart_static",
  ]
  deps = deps + tf_custom_op_library_additional_deps()
  if gpu_srcs:
    basename = name.split(".")[0]
    native.cc_library(
        name=basename + "_gpu",
        srcs=gpu_srcs,
        copts=_cuda_copts() + if_tensorrt(["-DGOOGLE_TENSORRT=1"]),
        features = if_cuda(["-use_header_modules"]),
        deps=deps + if_cuda(cuda_deps))
    cuda_deps.extend([":" + basename + "_gpu"])

  check_deps(
      name=name + "_check_deps",
      deps=deps + if_cuda(cuda_deps),
      disallowed_deps=[
          clean_dep("//tensorflow/core:framework"),
          clean_dep("//tensorflow/core:lib")
      ])
  tf_cc_shared_object(
      name=name,
      srcs=srcs,
      deps=deps + if_cuda(cuda_deps),
      data=[name + "_check_deps"],
      copts=tf_copts(is_external=True),
      features = ["windows_export_all_symbols"],
      linkopts=linkopts + select({
          "//conditions:default": [
              "-lm",
          ],
          clean_dep("//tensorflow:windows"): [],
          clean_dep("//tensorflow:windows_msvc"): [],
          clean_dep("//tensorflow:darwin"): [],
      }),)

register_extension_info(
    extension_name = "tf_custom_op_library",
    label_regex_for_dep = "{extension_name}",
)

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

register_extension_info(
    extension_name = "tf_custom_op_py_library",
    label_regex_for_dep = "{extension_name}",
)

# In tf_py_wrap_cc generated libraries
# module init functions are not exported unless
# they contain one of the keywords in the version file
# this prevents custom python modules.
# This function attempts to append init_module_name to list of
# exported functions in version script
def _append_init_to_versionscript_impl(ctx):
  mod_name = ctx.attr.module_name
  if ctx.attr.is_version_script:
    ctx.actions.expand_template(
      template=ctx.file.template_file,
      output=ctx.outputs.versionscript,
      substitutions={
        "global:":"global:\n     init_%s;\n     PyInit_*;"%(mod_name),
      },
      is_executable=False,
    )
  else:
    ctx.actions.expand_template(
      template=ctx.file.template_file,
      output=ctx.outputs.versionscript,
      substitutions={
        "*tensorflow*":"*tensorflow*\ninit_%s\nPyInit_*\n"%(mod_name),
      },
      is_executable=False,
    )


_append_init_to_versionscript= rule(
  implementation=_append_init_to_versionscript_impl,
  attrs={
    "module_name":attr.string(mandatory=True),
    "template_file":attr.label(allow_files=True,single_file=True,mandatory=True),
    "is_version_script":attr.bool(default=True,
      doc='whether target is a ld version script or exported symbol list',
      mandatory=False),
  },
  outputs={"versionscript":"%{name}.lds"},
)

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
  vscriptname=name+"_versionscript"
  _append_init_to_versionscript(
      name=vscriptname,
      module_name=module_name,
      is_version_script=select({
          "@local_config_cuda//cuda:darwin":False,
          "//conditions:default":True,
          }),
      template_file=select({
          "@local_config_cuda//cuda:darwin":clean_dep("//tensorflow:tf_exported_symbols.lds"),
          "//conditions:default":clean_dep("//tensorflow:tf_version_script.lds")
      })
  )
  extra_linkopts = select({
      "@local_config_cuda//cuda:darwin": [
          "-Wl,-exported_symbols_list",
          "$(location %s.lds)"%vscriptname,
      ],
      clean_dep("//tensorflow:windows"): [],
      clean_dep("//tensorflow:windows_msvc"): [],
      "//conditions:default": [
          "-Wl,--version-script",
          "$(location %s.lds)"%vscriptname,
      ]
  })
  extra_deps += select({
      "@local_config_cuda//cuda:darwin": [
          "%s.lds"%vscriptname,
      ],
      clean_dep("//tensorflow:windows"): [],
      clean_dep("//tensorflow:windows_msvc"): [],
      "//conditions:default": [
          "%s.lds"%vscriptname,
      ]
  })

  tf_cc_shared_object(
      name=cc_library_name,
      srcs=[module_name + ".cc"],
      copts=copts + if_not_windows([
          "-Wno-self-assign", "-Wno-sign-compare", "-Wno-write-strings"
      ]),
      linkopts=extra_linkopts,
      linkstatic=1,
      deps=deps + extra_deps,
      **kwargs)
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
          clean_dep("//tensorflow:windows"): [":" + cc_library_pyd_name],
          "//conditions:default": [":" + cc_library_name],
      }))

# This macro is for running python tests against system installed pip package
# on Windows.
#
# py_test is built as an executable python zip file on Windows, which contains all
# dependencies of the target. Because of the C++ extensions, it would be very
# inefficient if the py_test zips all runfiles, plus we don't need them when running
# tests against system installed pip package. So we'd like to get rid of the deps
# of py_test in this case.
#
# In order to trigger the tests without bazel clean after getting rid of deps,
# we introduce the following :
# 1. When --define=no_tensorflow_py_deps=true, the py_test depends on a marker
#    file of the pip package, the test gets to rerun when the pip package change.
#    Note that this only works on Windows. See the definition of
#    //third_party/tensorflow/tools/pip_package:win_pip_package_marker for specific reasons.
# 2. When --define=no_tensorflow_py_deps=false (by default), it's a normal py_test.
def py_test(deps=[], data=[], **kwargs):
  native.py_test(
      # TODO(jlebar): Ideally we'd use tcmalloc here.,
      deps=select({
          "//conditions:default": deps,
          clean_dep("//tensorflow:no_tensorflow_py_deps"): [],
      }),
      data = data + select({
          "//conditions:default": [],
          clean_dep("//tensorflow:no_tensorflow_py_deps"):
          ["//tensorflow/tools/pip_package:win_pip_package_marker"],
      }),
      **kwargs)

register_extension_info(
    extension_name = "py_test",
    label_regex_for_dep = "{extension_name}",
)

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
               xla_enabled=False,
               grpc_enabled=False):
  if xla_enabled:
    additional_deps = additional_deps + tf_additional_xla_deps_py()
  if grpc_enabled:
    additional_deps = additional_deps + tf_additional_grpc_deps_py()
  py_test(
      name=name,
      size=size,
      srcs=srcs,
      main=main,
      args=args,
      tags=tags,
      visibility=[clean_dep("//tensorflow:internal")],
      shard_count=shard_count,
      data=data,
      deps=[
            clean_dep("//tensorflow/python:extra_py_tests_deps"),
            clean_dep("//tensorflow/python:gradient_checker"),
          ] + additional_deps,
      flaky=flaky,
      srcs_version="PY2AND3")

register_extension_info(
    extension_name = "tf_py_test",
    label_regex_map = {"additional_deps": "deps:{extension_name}"},
)

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
                 xla_enabled=False,
                 grpc_enabled=False):
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
      xla_enabled=xla_enabled,
      grpc_enabled=grpc_enabled)

register_extension_info(
    extension_name = "cuda_py_test",
    label_regex_map = {"additional_deps": "additional_deps:{extension_name}"},
)

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
                 xla_enabled=False,
                 grpc_enabled=False):
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
      xla_enabled=xla_enabled,
      grpc_enabled=grpc_enabled)

register_extension_info(
    extension_name = "sycl_py_test",
    label_regex_map = {"additional_deps": "additional_deps:{extension_name}"},
)

def py_tests(name,
             srcs,
             size="medium",
             additional_deps=[],
             data=[],
             tags=[],
             shard_count=1,
             prefix="",
             xla_enabled=False,
             grpc_enabled=False):
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
        xla_enabled=xla_enabled,
        grpc_enabled=grpc_enabled)

def cuda_py_tests(name,
                  srcs,
                  size="medium",
                  additional_deps=[],
                  data=[],
                  shard_count=1,
                  tags=[],
                  prefix="",
                  xla_enabled=False,
                  grpc_enabled=False):
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
      xla_enabled=xla_enabled,
      grpc_enabled=grpc_enabled)

# Creates a genrule named <name> for running tools/proto_text's generator to
# make the proto_text functions, for the protos passed in <srcs>.
#
# Return a struct with fields (hdrs, srcs) containing the names of the
# generated files.
def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs, protodeps=[], deps=[], visibility=None):
  out_hdrs = (
      [p.replace(".proto", ".pb_text.h")
       for p in srcs] + [p.replace(".proto", ".pb_text-impl.h") for p in srcs])
  out_srcs = [p.replace(".proto", ".pb_text.cc") for p in srcs]
  native.genrule(
      name=name + "_srcs",
      srcs=srcs + protodeps + [clean_dep("//tensorflow/tools/proto_text:placeholder.txt")],
      outs=out_hdrs + out_srcs,
      visibility=visibility,
      cmd=
      "$(location //tensorflow/tools/proto_text:gen_proto_text_functions) "
      + "$(@D) " + srcs_relative_dir + " $(SRCS)",
      tools=[
          clean_dep("//tensorflow/tools/proto_text:gen_proto_text_functions")
      ],)

  native.filegroup(
      name=name + "_hdrs",
      srcs=out_hdrs,
      visibility=visibility,
  )

  native.cc_library(
      name=name,
      srcs=out_srcs,
      hdrs=out_hdrs,
      visibility=visibility,
      deps = deps,
  )

def tf_genrule_cmd_append_to_srcs(to_append):
  return ("cat $(SRCS) > $(@) && " + "echo >> $(@) && " + "echo " + to_append +
          " >> $(@)")

def tf_version_info_genrule():
  native.genrule(
      name="version_info_gen",
      srcs=[
          clean_dep("@local_config_git//:gen/spec.json"),
          clean_dep("@local_config_git//:gen/head"),
          clean_dep("@local_config_git//:gen/branch_ref"),
      ],
      outs=["util/version_info.cc"],
      cmd=
      "$(location //tensorflow/tools/git:gen_git_source.py) --generate $(SRCS) \"$@\" --git_tag_override=$${GIT_TAG_OVERRIDE:-}",
      local=1,
      tools=[clean_dep("//tensorflow/tools/git:gen_git_source.py")],)

def tf_py_build_info_genrule():
  native.genrule(
      name="py_build_info_gen",
      outs=["platform/build_info.py"],
      cmd=
     "$(location //tensorflow/tools/build_info:gen_build_info.py) --raw_generate \"$@\" --build_config " + if_cuda("cuda", "cpu"),
      local=1,
      tools=[clean_dep("//tensorflow/tools/build_info:gen_build_info.py")],)

def cc_library_with_android_deps(deps,
                                 android_deps=[],
                                 common_deps=[],
                                 copts=tf_copts(),
                                 **kwargs):
  deps = if_not_android(deps) + if_android(android_deps) + common_deps
  native.cc_library(deps=deps, copts=copts, **kwargs)

register_extension_info(
    extension_name = "cc_library_with_android_deps",
    label_regex_for_dep = "{extension_name}",
)
