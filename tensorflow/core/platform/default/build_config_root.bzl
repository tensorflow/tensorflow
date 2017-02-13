# Lower-level functionality for build config.
# The functions in this file might be referred by tensorflow.bzl. They have to
# be separate to avoid cyclic references.

WITH_XLA_SUPPORT = False

def tf_cuda_tests_tags():
  return ["local"]

def tf_sycl_tests_tags():
  return ["local"]

def tf_additional_plugin_deps():
  deps = []
  if WITH_XLA_SUPPORT:
    deps.append("//tensorflow/compiler/jit")
  return deps

def tf_additional_xla_deps_py():
  return []

def tf_additional_license_deps():
  licenses = []
  if WITH_XLA_SUPPORT:
    licenses.append("@llvm//:LICENSE.TXT")
  return licenses
