# Lower-level functionality for build config.
# The functions in this file might be referred by tensorflow.bzl. They have to
# be separate to avoid cyclic references.

load("@local_config_remote_execution//:remote_execution.bzl", "gpu_test_tags")

# RBE settings for tests that require a GPU. This is used in exec_properties of rules
# that need GPU access.
GPU_TEST_PROPERTIES = {
    "dockerRuntime": "nvidia",
    "Pool": "gpu-pool",
}

def tf_gpu_tests_tags():
    return ["requires-gpu", "gpu"] + gpu_test_tags()

# terminology changes: saving tf_cuda_* for compatibility
def tf_cuda_tests_tags():
    return tf_gpu_tests_tags()

def tf_exec_properties(kwargs):
    if ("tags" in kwargs and kwargs["tags"] != None and
        "remote-gpu" in kwargs["tags"]):
        return GPU_TEST_PROPERTIES
    return {}

def tf_additional_profiler_deps():
    return []

def tf_additional_xla_deps_py():
    return []

def tf_additional_grpc_deps_py():
    return []

def tf_additional_license_deps():
    return []

def tf_additional_tpu_ops_deps():
    return []

# TODO(vam): Remove, as it does not do anything anymore
def if_static(extra_deps, otherwise = [], macos = []):
    _ignore = otherwise  # unused argument
    _ignore = macos  # unused argument
    return extra_deps

def if_static_and_not_mobile(extra_deps, otherwise = []):
    return select({
        str(Label("//tsl:android")): otherwise,
        str(Label("//tsl:ios")): otherwise,
        "//conditions:default": extra_deps,
    })
