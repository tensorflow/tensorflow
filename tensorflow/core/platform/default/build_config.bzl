# Platform-specific build configurations.

load("@protobuf//:protobuf.bzl", "cc_proto_library")
load("@protobuf//:protobuf.bzl", "py_proto_library")
load("//tensorflow:tensorflow.bzl", "if_not_mobile")

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
                        protodeps = [], visibility = [], testonly = 0,
                        cc_libs = [],
                        cc_stubby_versions = None,
                        cc_grpc_version = None,
                        cc_api_version = 2, go_api_version = 2,
                        java_api_version = 2, py_api_version = 2,
                        js_api_version = 2, js_codegen = "jspb"):
  native.filegroup(
      name = name + "_proto_srcs",
      srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
      testonly = testonly,
  )

  use_grpc_plugin = None
  if cc_grpc_version:
    use_grpc_plugin = True
  cc_proto_library(
      name = name + "_cc",
      srcs = srcs,
      deps = tf_deps(protodeps, "_cc") + ["@protobuf//:cc_wkt_protos"],
      cc_libs = cc_libs + ["@protobuf//:protobuf"],
      copts = [
          "-Wno-unknown-warning-option",
          "-Wno-unused-but-set-variable",
          "-Wno-sign-compare",
      ],
      protoc = "@protobuf//:protoc",
      default_runtime = "@protobuf//:protobuf",
      use_grpc_plugin = use_grpc_plugin,
      testonly = testonly,
      visibility = visibility,
  )

def tf_proto_library_py(name, srcs=[], protodeps=[], deps=[], visibility=[],
                        testonly=0,
                        srcs_version="PY2AND3"):
  py_proto_library(
      name = name + "_py",
      srcs = srcs,
      srcs_version = srcs_version,
      deps = deps + tf_deps(protodeps, "_py") + ["@protobuf//:protobuf_python"],
      protoc = "@protobuf//:protoc",
      default_runtime = "@protobuf//:protobuf_python",
      visibility = visibility,
      testonly = testonly,
  )

def tf_proto_library(name, srcs = [], has_services = None,
                     protodeps = [], visibility = [], testonly = 0,
                     cc_libs = [],
                     cc_api_version = 2, go_api_version = 2,
                     java_api_version = 2, py_api_version = 2,
                     js_api_version = 2, js_codegen = "jspb"):
  """Make a proto library, possibly depending on other proto libraries."""
  tf_proto_library_cc(
      name = name,
      srcs = srcs,
      protodeps = protodeps,
      cc_libs = cc_libs,
      testonly = testonly,
      visibility = visibility,
  )

  tf_proto_library_py(
      name = name,
      srcs = srcs,
      protodeps = protodeps,
      srcs_version = "PY2AND3",
      testonly = testonly,
      visibility = visibility,
  )

def tf_additional_lib_hdrs(exclude = []):
  return select({
    "//tensorflow:windows" : native.glob([
        "platform/default/*.h",
        "platform/windows/*.h",
        "platform/posix/error.h",
      ], exclude = exclude),
    "//conditions:default" : native.glob([
        "platform/default/*.h",
        "platform/posix/*.h",
      ], exclude = exclude),
  })

def tf_additional_lib_srcs(exclude = []):
  return select({
    "//tensorflow:windows" : native.glob([
        "platform/default/*.cc",
        "platform/windows/*.cc",
        "platform/posix/error.cc",
      ], exclude = exclude),
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
  ]

def tf_additional_proto_srcs():
  return [
      "platform/default/logging.cc",
      "platform/default/protobuf.cc",
  ]

def tf_env_time_hdrs():
  return [
      "platform/env_time.h",
  ]

def tf_env_time_srcs():
  return select({
    "//tensorflow:windows" : native.glob([
        "platform/windows/env_time.cc",
        "platform/env_time.cc",
      ], exclude = []),
    "//conditions:default" : native.glob([
        "platform/posix/env_time.cc",
        "platform/env_time.cc",
      ], exclude = []),
  })

def tf_additional_stream_executor_srcs():
  return ["platform/default/stream_executor.h"]

def tf_additional_cupti_wrapper_deps():
  return ["//tensorflow/core/platform/default/gpu:cupti_wrapper"]

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
  return select({
      "//tensorflow:with_jemalloc": ["TENSORFLOW_USE_JEMALLOC"],
      "//conditions:default": [],
  })

def tf_additional_lib_deps():
  return select({
      "//tensorflow:with_jemalloc": ["@jemalloc"],
      "//conditions:default": [],
  })

def tf_additional_core_deps():
  return select({
      "//tensorflow:with_gcp_support": [
          "//tensorflow/core/platform/cloud:gcs_file_system",
      ],
      "//conditions:default": [],
  }) + select({
      "//tensorflow:with_hdfs_support": [
          "//tensorflow/core/platform/hadoop:hadoop_file_system",
      ],
      "//conditions:default": [],
  })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_op_deps():
  return select({
      "//tensorflow:windows": [],
      "//tensorflow:android": [],
      "//tensorflow:ios": [],
      "//tensorflow:with_gcp_support": [
        "//tensorflow/contrib/cloud:bigquery_reader_ops_op_lib",
      ],
      "//conditions:default": [],
  })

# TODO(jart, jhseu): Delete when GCP is default on.
def tf_additional_cloud_kernel_deps():
  return select({
      "//tensorflow:windows": [],
      "//tensorflow:android": [],
      "//tensorflow:ios": [],
      "//tensorflow:with_gcp_support": [
        "//tensorflow/contrib/cloud/kernels:bigquery_reader_ops",
      ],
      "//conditions:default": [],
  })

def tf_lib_proto_parsing_deps():
  return [
      ":protos_all_cc",
      "//tensorflow/core/platform/default/build_config:proto_parsing",
  ]
