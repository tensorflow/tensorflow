# Lower-level functionality for build config.
# The functions in this file might be referred by tensorflow.bzl. They have to
# be separate to avoid cyclic references.

def tf_cuda_tests_tags():
  return ["local"]

def tf_sycl_tests_tags():
  return ["local"]

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
