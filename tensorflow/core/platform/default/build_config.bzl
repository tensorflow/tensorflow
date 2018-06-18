# Platform-specific build configurations.

load("@protobuf_archive//:protobuf.bzl", "proto_gen")
load("//tensorflow:tensorflow.bzl", "if_not_mobile")
load("//tensorflow:tensorflow.bzl", "if_windows")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

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
    deps=[],
    py_deps=[],
    srcs=[],
    **kwargs):
  """Compiles a group of .pyx / .pxd / .py files.

  First runs Cython to create .cpp files for each input .pyx or .py + .pxd
  pair. Then builds a shared object for each, passing "deps" to each cc_binary
  rule (includes Python headers by default). Finally, creates a py_library rule
  with the shared objects and any pure Python "srcs", with py_deps as its
  dependencies; the shared objects can be imported like normal Python files.

  Args:
    name: Name for the rule.
    deps: C/C++ dependencies of the Cython (e.g. Numpy headers).
    py_deps: Pure Python dependencies of the final library.
    srcs: .py, .pyx, or .pxd files to either compile or pass through.
    **kwargs: Extra keyword arguments passed to the py_library.
  """
  # First filter out files that should be run compiled vs. passed through.
  py_srcs = []
  pyx_srcs = []
  pxd_srcs = []
  for src in srcs:
    if src.endswith(".pyx") or (src.endswith(".py")
                                and src[:-3] + ".pxd" in srcs):
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
        cmd = "PYTHONHASHSEED=0 $(location @cython//:cython_binary) --cplus $(SRCS) --output-file $(OUTS)",
        tools = ["@cython//:cython_binary"] + pxd_srcs,
    )

  shared_objects = []
  for src in pyx_srcs:
    stem = src.split(".")[0]
    shared_object_name = stem + ".so"
    native.cc_binary(
        name=shared_object_name,
        srcs=[stem + ".cpp"],
        deps=deps + ["//third_party/python_runtime:headers"],
        linkshared = 1,
    )
    shared_objects.append(shared_object_name)

  # Now create a py_library with these shared objects as data.
  native.py_library(
      name=name,
      srcs=py_srcs,
      deps=py_deps,
      srcs_version = "PY2AND3",
      data=shared_objects,
      **kwargs
  )

def _proto_cc_hdrs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
  return ret

def _proto_cc_srcs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
  return ret

def _proto_py_outs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + "_pb2.py" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + "_pb2_grpc.py" for s in srcs]
  return ret

# Re-defined protocol buffer rule to allow building "header only" protocol
# buffers, to avoid duplicate registrations. Also allows non-iterable cc_libs
# containing select() statements.
def cc_proto_library(
    name,
    srcs=[],
    deps=[],
    cc_libs=[],
    include=None,
    protoc="@protobuf_archive//:protoc",
    internal_bootstrap_hack=False,
    use_grpc_plugin=False,
    use_grpc_namespace=False,
    default_header=False,
    **kargs):
  """Bazel rule to create a C++ protobuf library from proto source files.

  Args:
    name: the name of the cc_proto_library.
    srcs: the .proto files of the cc_proto_library.
    deps: a list of dependency labels; must be cc_proto_library.
    cc_libs: a list of other cc_library targets depended by the generated
        cc_library.
    include: a string indicating the include path of the .proto files.
    protoc: the label of the protocol compiler to generate the sources.
    internal_bootstrap_hack: a flag indicate the cc_proto_library is used only
        for bootstraping. When it is set to True, no files will be generated.
        The rule will simply be a provider for .proto files, so that other
        cc_proto_library can depend on it.
    use_grpc_plugin: a flag to indicate whether to call the grpc C++ plugin
        when processing the proto files.
    default_header: Controls the naming of generated rules. If True, the `name`
        rule will be header-only, and an _impl rule will contain the
        implementation. Otherwise the header-only rule (name + "_headers_only")
        must be referred to explicitly.
    **kargs: other keyword arguments that are passed to cc_library.
  """

  includes = []
  if include != None:
    includes = [include]

  if internal_bootstrap_hack:
    # For pre-checked-in generated files, we add the internal_bootstrap_hack
    # which will skip the codegen action.
    proto_gen(
        name=name + "_genproto",
        srcs=srcs,
        deps=[s + "_genproto" for s in deps],
        includes=includes,
        protoc=protoc,
        visibility=["//visibility:public"],
    )
    # An empty cc_library to make rule dependency consistent.
    native.cc_library(
        name=name,
        **kargs)
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
      name=name + "_genproto",
      srcs=srcs,
      deps=[s + "_genproto" for s in deps],
      includes=includes,
      protoc=protoc,
      plugin=grpc_cpp_plugin,
      plugin_language="grpc",
      plugin_options=plugin_options,
      gen_cc=1,
      outs=outs,
      visibility=["//visibility:public"],
  )

  if use_grpc_plugin:
    cc_libs += ["//external:grpc_lib"]

  if default_header:
    header_only_name = name
    impl_name = name + "_impl"
  else:
    header_only_name = name + "_headers_only"
    impl_name = name

  native.cc_library(
      name=impl_name,
      srcs=gen_srcs,
      hdrs=gen_hdrs,
      deps=cc_libs + deps,
      includes=includes,
      **kargs)
  native.cc_library(
      name=header_only_name,
      deps=["@protobuf_archive//:protobuf_headers"] + if_static([impl_name]),
      hdrs=gen_hdrs,
      **kargs)

# Re-defined protocol buffer rule to bring in the change introduced in commit
# https://github.com/google/protobuf/commit/294b5758c373cbab4b72f35f4cb62dc1d8332b68
# which was not part of a stable protobuf release in 04/2018.
# TODO(jsimsa): Remove this once the protobuf dependency version is updated
# to include the above commit.
def py_proto_library(
        name,
        srcs=[],
        deps=[],
        py_libs=[],
        py_extra_srcs=[],
        include=None,
        default_runtime="@protobuf_archive//:protobuf_python",
        protoc="@protobuf_archive//:protoc",
        use_grpc_plugin=False,
        **kargs):
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
    **kargs: other keyword arguments that are passed to cc_library.
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

  proto_gen(
      name=name + "_genproto",
      srcs=srcs,
      deps=[s + "_genproto" for s in deps],
      includes=includes,
      protoc=protoc,
      gen_py=1,
      outs=outs,
      visibility=["//visibility:public"],
      plugin=grpc_python_plugin,
      plugin_language="grpc"
  )

  if default_runtime and not default_runtime in py_libs + deps:
    py_libs = py_libs + [default_runtime]

  native.py_library(
      name=name,
      srcs=outs+py_extra_srcs,
      deps=py_libs+deps,
      imports=includes,
      **kargs)

def tf_proto_library_cc(name, srcs = [], has_services = None,
                        protodeps = [],
                        visibility = [], testonly = 0,
                        cc_libs = [],
                        cc_stubby_versions = None,
                        cc_grpc_version = None,
                        j2objc_api_version = 1,
                        cc_api_version = 2,
                        dart_api_version = 2,
                        java_api_version = 2, py_api_version = 2,
                        js_api_version = 2, js_codegen = "jspb",
                        default_header = False):
  js_codegen = js_codegen  # unused argument
  js_api_version = js_api_version  # unused argument
  native.filegroup(
      name = name + "_proto_srcs",
      srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
      testonly = testonly,
      visibility = visibility,
  )

  use_grpc_plugin = None
  if cc_grpc_version:
    use_grpc_plugin = True

  cc_deps = tf_deps(protodeps, "_cc")
  cc_name = name + "_cc"
  if not srcs:
    # This is a collection of sub-libraries. Build header-only and impl
    # libraries containing all the sources.
    proto_gen(
        name = cc_name + "_genproto",
        deps = [s + "_genproto" for s in cc_deps],
        protoc = "@protobuf_archive//:protoc",
        visibility=["//visibility:public"],
    )
    native.cc_library(
        name = cc_name,
        deps = cc_deps + ["@protobuf_archive//:protobuf_headers"] +
               if_static([name + "_cc_impl"]),
        testonly = testonly,
        visibility = visibility,
    )
    native.cc_library(
        name = cc_name + "_impl",
        deps = [s + "_impl" for s in cc_deps] + ["@protobuf_archive//:cc_wkt_protos"],
    )

    return

  cc_proto_library(
      name = cc_name,
      srcs = srcs,
      deps = cc_deps + ["@protobuf_archive//:cc_wkt_protos"],
      cc_libs = cc_libs + if_static(
          ["@protobuf_archive//:protobuf"],
          ["@protobuf_archive//:protobuf_headers"]
      ),
      copts = if_not_windows([
          "-Wno-unknown-warning-option",
          "-Wno-unused-but-set-variable",
          "-Wno-sign-compare",
      ]),
      protoc = "@protobuf_archive//:protoc",
      use_grpc_plugin = use_grpc_plugin,
      testonly = testonly,
      visibility = visibility,
      default_header = default_header,
  )

def tf_proto_library_py(name, srcs=[], protodeps=[], deps=[], visibility=[],
                        testonly=0, srcs_version="PY2AND3", use_grpc_plugin=False):
  py_deps = tf_deps(protodeps, "_py")
  py_name = name + "_py"
  if not srcs:
    # This is a collection of sub-libraries. Build header-only and impl
    # libraries containing all the sources.
    proto_gen(
        name = py_name + "_genproto",
        deps = [s + "_genproto" for s in py_deps],
        protoc = "@protobuf_archive//:protoc",
        visibility=["//visibility:public"],
    )
    native.py_library(
        name = py_name,
        deps = py_deps + ["@protobuf_archive//:protobuf_python"],
        testonly = testonly,
        visibility = visibility,
    )
    return

  py_proto_library(
      name = py_name,
      srcs = srcs,
      srcs_version = srcs_version,
      deps = deps + py_deps + ["@protobuf_archive//:protobuf_python"],
      protoc = "@protobuf_archive//:protoc",
      default_runtime = "@protobuf_archive//:protobuf_python",
      visibility = visibility,
      testonly = testonly,
      use_grpc_plugin = use_grpc_plugin,
  )

def tf_jspb_proto_library(**kwargs):
  pass

def tf_nano_proto_library(**kwargs):
  pass

def tf_proto_library(name, srcs = [], has_services = None,
                     protodeps = [],
                     visibility = [], testonly = 0,
                     cc_libs = [],
                     cc_api_version = 2, cc_grpc_version = None,
                     dart_api_version = 2, j2objc_api_version = 1,
                     java_api_version = 2, py_api_version = 2,
                     js_api_version = 2, js_codegen = "jspb",
                     provide_cc_alias = False,
                     default_header = False):
  """Make a proto library, possibly depending on other proto libraries."""
  _ignore = (js_api_version, js_codegen, provide_cc_alias)

  tf_proto_library_cc(
      name = name,
      srcs = srcs,
      protodeps = protodeps,
      cc_grpc_version = cc_grpc_version,
      cc_libs = cc_libs,
      testonly = testonly,
      visibility = visibility,
      default_header = default_header,
  )

  tf_proto_library_py(
      name = name,
      srcs = srcs,
      protodeps = protodeps,
      srcs_version = "PY2AND3",
      testonly = testonly,
      visibility = visibility,
      use_grpc_plugin = has_services,
  )

# A list of all files under platform matching the pattern in 'files'. In
# contrast with 'tf_platform_srcs' below, which seletive collects files that
# must be compiled in the 'default' platform, this is a list of all headers
# mentioned in the platform/* files.
def tf_platform_hdrs(files):
  return native.glob(["platform/*/" + f for f in files])

def tf_platform_srcs(files):
  base_set = ["platform/default/" + f for f in files]
  windows_set = base_set + ["platform/windows/" + f for f in files]
  posix_set = base_set + ["platform/posix/" + f for f in files]

  # Handle cases where we must also bring the posix file in. Usually, the list
  # of files to build on windows builds is just all the stuff in the
  # windows_set. However, in some cases the implementations in 'posix/' are
  # just what is necessary and historically we choose to simply use the posix
  # file instead of making a copy in 'windows'.
  for f in files:
    if f == "error.cc":
      windows_set.append("platform/posix/" + f)

  return select({
    "//tensorflow:windows" : native.glob(windows_set),
    "//tensorflow:windows_msvc" : native.glob(windows_set),
    "//conditions:default" : native.glob(posix_set),
  })

def tf_additional_lib_hdrs(exclude = []):
  windows_hdrs = native.glob([
      "platform/default/*.h",
      "platform/windows/*.h",
      "platform/posix/error.h",
  ], exclude = exclude)
  return select({
    "//tensorflow:windows" : windows_hdrs,
    "//tensorflow:windows_msvc" : windows_hdrs,
    "//conditions:default" : native.glob([
        "platform/default/*.h",
        "platform/posix/*.h",
      ], exclude = exclude),
  })

def tf_additional_lib_srcs(exclude = []):
  windows_srcs = native.glob([
      "platform/default/*.cc",
      "platform/windows/*.cc",
      "platform/posix/error.cc",
  ], exclude = exclude)
  return select({
    "//tensorflow:windows" : windows_srcs,
    "//tensorflow:windows_msvc" : windows_srcs,
    "//conditions:default" : native.glob([
        "platform/default/*.cc",
        "platform/posix/*.cc",
      ], exclude = exclude),
  })

def tf_additional_minimal_lib_srcs():
  return [
      "platform/default/integral_types.h",
      "platform/default/mutex.h",
  ]

def tf_additional_proto_hdrs():
  return [
      "platform/default/integral_types.h",
      "platform/default/logging.h",
      "platform/default/protobuf.h"
  ] + if_windows([
      "platform/windows/integral_types.h",
  ])

def tf_additional_proto_srcs():
  return [
      "platform/default/protobuf.cc",
  ]

def tf_additional_human_readable_json_deps():
  return []

def tf_additional_all_protos():
  return ["//tensorflow/core:protos_all"]

def tf_protos_all_impl():
  return ["//tensorflow/core:protos_all_cc_impl"]

def tf_protos_all():
  return if_static(
      extra_deps=tf_protos_all_impl(),
      otherwise=["//tensorflow/core:protos_all_cc"])

def tf_protos_grappler_impl():
  return ["//tensorflow/core/grappler/costs:op_performance_data_cc_impl"]

def tf_protos_grappler():
  return if_static(
      extra_deps=tf_protos_grappler_impl(),
      otherwise=["//tensorflow/core/grappler/costs:op_performance_data_cc"])

def tf_additional_cupti_wrapper_deps():
  return ["//tensorflow/core/platform/default/gpu:cupti_wrapper"]

def tf_additional_device_tracer_srcs():
  return ["platform/default/device_tracer.cc"]

def tf_additional_device_tracer_cuda_deps():
  return []

def tf_additional_device_tracer_deps():
  return []

def tf_additional_libdevice_data():
  return []

def tf_additional_libdevice_deps():
  return ["@local_config_cuda//cuda:cuda_headers"]

def tf_additional_libdevice_srcs():
  return ["platform/default/cuda_libdevice_path.cc"]

def tf_additional_test_deps():
  return []

def tf_additional_test_srcs():
  return [
      "platform/default/test_benchmark.cc",
  ] + select({
      "//tensorflow:windows" : [
          "platform/windows/test.cc"
        ],
      "//conditions:default" : [
          "platform/posix/test.cc",
        ],
    })

def tf_kernel_tests_linkstatic():
  return 0

def tf_additional_lib_defines():
  """Additional defines needed to build TF libraries."""
  return select({
      "//tensorflow:with_jemalloc_linux_x86_64": ["TENSORFLOW_USE_JEMALLOC"],
      "//tensorflow:with_jemalloc_linux_ppc64le":["TENSORFLOW_USE_JEMALLOC"],
      "//conditions:default": [],
  }) + if_not_mobile(["TENSORFLOW_USE_ABSL"])

def tf_additional_lib_deps():
  """Additional dependencies needed to build TF libraries."""
  return if_not_mobile(["@com_google_absl//absl/base:base"]) + if_static(
      ["@nsync//:nsync_cpp"],
      ["@nsync//:nsync_headers"]
  ) + select({
      "//tensorflow:with_jemalloc_linux_x86_64_dynamic": ["@jemalloc//:jemalloc_headers"],
      "//tensorflow:with_jemalloc_linux_ppc64le_dynamic": ["@jemalloc//:jemalloc_headers"],
      "//tensorflow:with_jemalloc_linux_x86_64": ["@jemalloc//:jemalloc_impl"],
      "//tensorflow:with_jemalloc_linux_ppc64le": ["@jemalloc//:jemalloc_impl"],
      "//conditions:default": [],
  })

def tf_additional_core_deps():
  return select({
      "//tensorflow:with_gcp_support_android_override": [],
      "//tensorflow:with_gcp_support_ios_override": [],
      "//tensorflow:with_gcp_support": [
          "//tensorflow/core/platform/cloud:gcs_file_system",
      ],
      "//conditions:default": [],
  }) + select({
      "//tensorflow:with_hdfs_support_windows_override": [],
      "//tensorflow:with_hdfs_support_android_override": [],
      "//tensorflow:with_hdfs_support_ios_override": [],
      "//tensorflow:with_hdfs_support": [
          "//tensorflow/core/platform/hadoop:hadoop_file_system",
      ],
      "//conditions:default": [],
  }) + select({
      "//tensorflow:with_s3_support_windows_override": [],
      "//tensorflow:with_s3_support_android_override": [],
      "//tensorflow:with_s3_support_ios_override": [],
      "//tensorflow:with_s3_support": [
          "//tensorflow/core/platform/s3:s3_file_system",
      ],
      "//conditions:default": [],
  })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_op_deps():
  return select({
      "//tensorflow:with_gcp_support_windows_override": [],
      "//tensorflow:with_gcp_support_android_override": [],
      "//tensorflow:with_gcp_support_ios_override": [],
      "//tensorflow:with_gcp_support": [
        "//tensorflow/contrib/cloud:bigquery_reader_ops_op_lib",
        "//tensorflow/contrib/cloud:gcs_config_ops_op_lib",
      ],
      "//conditions:default": [],
  })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_kernel_deps():
  return select({
      "//tensorflow:with_gcp_support_windows_override": [],
      "//tensorflow:with_gcp_support_android_override": [],
      "//tensorflow:with_gcp_support_ios_override": [],
      "//tensorflow:with_gcp_support": [
        "//tensorflow/contrib/cloud/kernels:bigquery_reader_ops",
        "//tensorflow/contrib/cloud/kernels:gcs_config_ops",
      ],
      "//conditions:default": [],
  })

def tf_lib_proto_parsing_deps():
  return [
      ":protos_all_cc",
      "//third_party/eigen3",
      "//tensorflow/core/platform/default/build_config:proto_parsing",
  ]

def tf_additional_verbs_lib_defines():
  return select({
      "//tensorflow:with_verbs_support": ["TENSORFLOW_USE_VERBS"],
      "//conditions:default": [],
  })

def tf_additional_mpi_lib_defines():
  return select({
      "//tensorflow:with_mpi_support": ["TENSORFLOW_USE_MPI"],
      "//conditions:default": [],
  })

def tf_additional_gdr_lib_defines():
  return select({
      "//tensorflow:with_gdr_support": ["TENSORFLOW_USE_GDR"],
      "//conditions:default": [],
  })

def tf_py_clif_cc(name, visibility=None, **kwargs):
  pass

def tf_pyclif_proto_library(name, proto_lib, proto_srcfile="", visibility=None,
                            **kwargs):
  pass

def tf_additional_binary_deps():
  return ["@nsync//:nsync_cpp"] + if_cuda(
      [
          "//tensorflow/stream_executor:cuda_platform",
          "//tensorflow/core/platform/default/build_config:cuda",
      ],
  ) + select({
      "//tensorflow:with_jemalloc_linux_x86_64": ["@jemalloc//:jemalloc_impl"],
      "//tensorflow:with_jemalloc_linux_ppc64le": ["@jemalloc//:jemalloc_impl"],
      "//conditions:default": [],
  })  + [
      # TODO(allenl): Split these out into their own shared objects (they are
      # here because they are shared between contrib/ op shared objects and
      # core).
      "//tensorflow/core/kernels:lookup_util",
      "//tensorflow/core/util/tensor_bundle",
  ] + if_mkl(
      [
          "//third_party/mkl:intel_binary_blob",
      ],
  )
