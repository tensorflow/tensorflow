"""Lower-level functionality for build config.

The functions in this file might be referred by tensorflow.bzl. They have to
be separate to avoid cyclic references.
"""

load("@local_config_remote_execution//:remote_execution.bzl", "gpu_test_tags")
load("@local_config_rocm//rocm:build_defs.bzl", "is_rocm_configured")
load("@xla//third_party/py/rules_pywrap:pywrap.default.bzl", "use_pywrap_rules")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "is_cuda_configured")

visibility(DEFAULT_LOAD_VISIBILITY)

# RBE settings for tests that require a GPU. This is used in exec_properties of rules
# that need GPU access.
GPU_TEST_PROPERTIES = {
    "dockerRuntime": "nvidia",
    "Pool": "gpu-pool",
}

def tf_gpu_tests_tags():
    """Gets tags for TensorFlow GPU tests based on the configured environment.

    Returns:
        A list of tags to be added to the test target.
    """
    if is_cuda_configured():
        return ["requires-gpu-cuda", "gpu"] + gpu_test_tags()
    elif is_rocm_configured():
        return ["requires-gpu-rocm", "gpu"] + gpu_test_tags()
    else:
        # If neither CUDA nor ROCm is configured, we assume no GPU support.
        # This is a fallback and should not be used in practice.
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

# TODO(b/356020232): remove completely after migration is done
# Include specific extra dependencies when building statically, or
# another set of dependencies otherwise. If "macos" is provided, that
# dependency list is used when using the framework_shared_object config
# on MacOS platforms. If "macos" is not provided, the "otherwise" list is
# used for all framework_shared_object platforms including MacOS.
# buildifier: disable=function-docstring
def if_static(extra_deps, otherwise = [], macos = []):
    if use_pywrap_rules():
        return extra_deps

    ret = {
        str(Label("//xla/tsl:framework_shared_object")): otherwise,
        "//conditions:default": extra_deps,
    }
    if macos:
        ret[str(Label("//xla/tsl:macos_with_framework_shared_object"))] = macos
    return select(ret)

# TODO(b/356020232): remove completely after migration is done
def if_static_and_not_mobile(extra_deps, otherwise = []):
    if use_pywrap_rules():
        return extra_deps

    return select({
        str(Label("//xla/tsl:framework_shared_object")): otherwise,
        str(Label("//xla/tsl:android")): otherwise,
        str(Label("//xla/tsl:ios")): otherwise,
        "//conditions:default": extra_deps,
    })

# TODO(b/356020232): remove completely after migration is done
def if_pywrap(if_true = [], if_false = []):
    return if_true if use_pywrap_rules() else if_false

def if_llvm_aarch32_available(then, otherwise = []):
    return select({
        str(Label("//xla/tsl:aarch32_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_aarch64_available(then, otherwise = []):
    return select({
        str(Label("//xla/tsl:aarch64_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_arm_available(then, otherwise = []):
    return select({
        str(Label("//xla/tsl:arm_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_hexagon_available(then, otherwise = []):
    _ = then  # @unused
    return otherwise

def if_llvm_powerpc_available(then, otherwise = []):
    return select({
        str(Label("//xla/tsl:ppc64le_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_system_z_available(then, otherwise = []):
    return select({
        str(Label("//xla/tsl:s390x_or_cross")): then,
        "//conditions:default": otherwise,
    })

def if_llvm_x86_available(then, otherwise = []):
    return select({
        str(Label("//xla/tsl:x86_or_cross")): then,
        "//conditions:default": otherwise,
    })
