# Lower-level functionality for build config.
# The functions in this file might be referred by tensorflow.bzl. They have to
# be separate to avoid cyclic references.

def tf_cuda_tests_tags():
  return ["requires-gpu"]

def tf_sycl_tests_tags():
  return ["requires-gpu"]

def tf_additional_plugin_deps():
  return select({
      "//tensorflow:with_xla_support": ["//tensorflow/compiler/jit"],
      "//conditions:default": [],
  })

def tf_additional_xla_deps_py():
  return []

def tf_additional_license_deps():
  return select({
      "//tensorflow:with_xla_support": ["@llvm//:LICENSE.TXT"],
      "//conditions:default": [],
  })

def tf_additional_verbs_deps():
  return select({
      "//tensorflow:with_verbs_support": [
          "//tensorflow/contrib/verbs:verbs_server_lib",
          "//tensorflow/contrib/verbs:grpc_verbs_client",
      ], 
      "//conditions:default": [],
  })

def tf_additional_mpi_deps():
  return select({
      "//tensorflow:with_mpi_support": [
          "//tensorflow/contrib/mpi:mpi_server_lib",
      ],
      "//conditions:default": [],
  })

def tf_additional_gdr_deps():
  return select({
      "//tensorflow:with_gdr_support": [
          "//tensorflow/contrib/gdr:gdr_server_lib",
      ],
      "//conditions:default": [],
  })
