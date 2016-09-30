# Platform-specific build configurations.

load("@protobuf//:protobuf.bzl", "cc_proto_library")
load("@protobuf//:protobuf.bzl", "py_proto_library")

# configure may change the following lines to True
WITH_GCP_SUPPORT = False
WITH_HDFS_SUPPORT = False

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

def tf_proto_library_cc(name, srcs = [], has_services = None,
                        deps = [], visibility = [], testonly = 0,
                        cc_libs = [],
                        cc_stubby_versions = None,
                        cc_grpc_version = None,
                        cc_api_version = 2, go_api_version = 2,
                        java_api_version = 2,
                        py_api_version = 2):
  native.filegroup(
      name = name + "_proto_srcs",
      srcs = srcs + tf_deps(deps, "_proto_srcs"),
      testonly = testonly,
  )

  use_grpc_plugin = None
  if cc_grpc_version:
    use_grpc_plugin = True
  cc_proto_library(
      name = name + "_cc",
      srcs = srcs + tf_deps(deps, "_proto_srcs"),
      deps = deps + ["@protobuf//:cc_wkt_protos"],
      cc_libs = cc_libs + ["@protobuf//:protobuf"],
      protoc = "@protobuf//:protoc",
      default_runtime = "@protobuf//:protobuf",
      use_grpc_plugin = use_grpc_plugin,
      testonly = testonly,
      visibility = visibility,
  )

def tf_proto_library_py(name, srcs=[], deps=[], visibility=[], testonly=0,
                        srcs_version="PY2AND3"):
  py_proto_library(
      name = name + "_py",
      srcs = srcs,
      srcs_version = srcs_version,
      deps = deps,
      protoc = "@protobuf//:protoc",
      default_runtime = "@protobuf//:protobuf_python",
      visibility = visibility,
      testonly = testonly,
  )

def tf_proto_library(name, srcs = [], has_services = None,
                     deps = [], visibility = [], testonly = 0,
                     cc_libs = [],
                     cc_api_version = 2, go_api_version = 2,
                     java_api_version = 2,
                     py_api_version = 2):
  tf_proto_library_cc(
      name = name,
      srcs = srcs + tf_deps(deps, "_proto_srcs"),
      deps = deps,
      cc_libs = cc_libs,
      testonly = testonly,
      visibility = visibility,
  )

  tf_proto_library_py(
      name = name,
      srcs = srcs + tf_deps(deps, "_proto_srcs"),
      srcs_version = "PY2AND3",
      deps = deps + ["@protobuf//:protobuf_python"],
      testonly = testonly,
      visibility = visibility,
  )

def tf_additional_lib_hdrs():
  return [
      "platform/default/*.h",
      "platform/posix/*.h",
  ]

def tf_additional_lib_srcs():
  return [
      "platform/default/*.cc",
      "platform/posix/*.cc",
  ]

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
  ]

def tf_additional_proto_srcs():
  return [
      "platform/default/logging.cc",
      "platform/default/protobuf.cc",
  ]

def tf_additional_stream_executor_srcs():
  return ["platform/default/stream_executor.h"]

def tf_additional_cupti_wrapper_deps():
  return ["//tensorflow/core/platform/default/gpu:cupti_wrapper"]

def tf_additional_test_deps():
  return []

def tf_additional_test_srcs():
  return ["platform/default/test_benchmark.cc", "platform/posix/test.cc"]

def tf_kernel_tests_linkstatic():
  return 0

def tf_additional_lib_deps():
  deps = []
  if WITH_GCP_SUPPORT:
    deps.append("//tensorflow/core/platform/cloud:gcs_file_system")
  if WITH_HDFS_SUPPORT:
    deps.append("//tensorflow/core/platform/hadoop:hadoop_file_system")
  return deps
