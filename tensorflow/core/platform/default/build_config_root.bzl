# Lower-level functionality for build config.
# The functions in this file might be referred by tensorflow.bzl. They have to
# be separate to avoid cyclic references.

def tf_cuda_tests_tags():
  return ["local"]

def tf_sycl_tests_tags():
  return ["local"]
