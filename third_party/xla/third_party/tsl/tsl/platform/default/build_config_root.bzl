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

# Include specific extra dependencies when building statically, or
# another set of dependencies otherwise. If "macos" is provided, that
# dependency list is used when using the framework_shared_object config
# on MacOS platforms. If "macos" is not provided, the "otherwise" list is
# used for all framework_shared_object platforms including MacOS.
def if_static(extra_deps, otherwise = [], macos = []):
    ret = {
        str(Label("@local_xla//xla/tsl:framework_shared_object")): otherwise,
        "//conditions:default": extra_deps,
    }
    if macos:
        ret[str(Label("@local_xla//xla/tsl:macos_with_framework_shared_object"))] = macos
    return select(ret)

def if_static_and_not_mobile(extra_deps, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:framework_shared_object")): otherwise,
        str(Label("@local_xla//xla/tsl:android")): otherwise,
        str(Label("@local_xla//xla/tsl:ios")): otherwise,
        "//conditions:default": extra_deps,
    })

def if_llvm_aarch32_available(then, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:aarch32_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_aarch64_available(then, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:aarch64_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_arm_available(then, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:arm_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_hexagon_available(then, otherwise = []):
    return otherwise

def if_llvm_powerpc_available(then, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:ppc64le_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_system_z_available(then, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:s390x_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_x86_available(then, otherwise = []):
    return select({
        str(Label("@local_xla//xla/tsl:x86_or_cross")): then,
        "//conditions:default": otherwise,
    })
