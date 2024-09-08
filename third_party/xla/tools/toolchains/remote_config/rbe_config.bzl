"""Macro that creates external repositories for remote config."""

load("//third_party/gpus:rocm_configure.bzl", "remote_rocm_configure")
load("//third_party/py:python_configure.bzl", "local_python_configure", "remote_python_configure")
load("//third_party/remote_config:remote_platform_configure.bzl", "remote_platform_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "remote_tensorrt_configure")
load("//tools/toolchains/remote_config:containers.bzl", "containers")

def _container_image_uri(container_name):
    container = containers[container_name]
    return "docker://%s/%s@%s" % (container["registry"], container["repository"], container["digest"])

def _tensorflow_rbe_config(name, compiler, os, rocm_version = None, cuda_version = None, cudnn_version = None, tensorrt_version = None, tensorrt_install_path = None, compiler_prefix = None):
    if cuda_version != None and rocm_version != None:
        fail("Specifying both cuda_version and rocm_version is not supported.")

    env = {
        "ABI_VERSION": "gcc",
        "ABI_LIBC_VERSION": "glibc_2.19",
        "BAZEL_COMPILER": compiler,
        "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
        "BAZEL_TARGET_LIBC": "glibc_2.19",
        "BAZEL_TARGET_CPU": "k8",
        "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
        "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
        "CC": compiler,
        "CLEAR_CACHE": "1",
        "HOST_CXX_COMPILER": compiler,
        "HOST_C_COMPILER": compiler,
    }

    if cuda_version != None:
        # The cuda toolchain currently contains its own C++ toolchain definition,
        # so we do not fetch local_config_cc.
        env.update({
            "TF_ENABLE_XLA": "1",
            "TF_NEED_TENSORRT": "0",
            "TF_TENSORRT_VERSION": tensorrt_version if tensorrt_version != None else "",
            "TENSORRT_INSTALL_PATH": tensorrt_install_path if tensorrt_install_path != None else "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": compiler if not compiler.endswith("clang") else "",
            "GCC_HOST_COMPILER_PREFIX": compiler_prefix if compiler_prefix != None else "/usr/bin",
        })

        cuda_version_in_container = ".".join(cuda_version.split(".")[:2])
        cudnn_version_in_container = ".".join(cudnn_version.split(".")[:2])
        container_name = "cuda%s-cudnn%s-%s" % (
            cuda_version_in_container,
            cudnn_version_in_container,
            os,
        )
        container_image = _container_image_uri(container_name)
        exec_properties = {
            "container-image": container_image,
            "Pool": "default",
        }

        remote_tensorrt_configure(
            name = "%s_config_tensorrt" % name,
            environ = env,
            exec_properties = exec_properties,
        )
    elif rocm_version != None:
        # The rocm toolchain currently contains its own C++ toolchain definition,
        # so we do not fetch local_config_cc.
        env.update({
            "TF_NEED_ROCM": "1",
            "TF_ENABLE_XLA": "0",
        })

        container_name = "rocm-%s" % (os)
        container_image = _container_image_uri(container_name)
        exec_properties = {
            "container-image": container_image,
            "Pool": "default",
        }

        remote_rocm_configure(
            name = "%s_config_rocm" % name,
            environ = env,
            exec_properties = exec_properties,
        )

    else:
        fail("Neither cuda_version nor rocm_version specified.")

    remote_platform_configure(
        name = "%s_config_platform" % name,
        platform = "linux",
        platform_exec_properties = exec_properties,
    )

def _tensorflow_rbe_win_config(name, python_bin_path, container_name = "windows-1803"):
    container_image = _container_image_uri(container_name)
    exec_properties = {
        "container-image": container_image,
        "OSFamily": "Windows",
    }

    env = {
        "PYTHON_BIN_PATH": python_bin_path,
    }

    remote_platform_configure(
        name = "%s_config_platform" % name,
        platform = "windows",
        platform_exec_properties = exec_properties,
    )

    remote_python_configure(
        name = "%s_config_python" % name,
        exec_properties = exec_properties,
        environ = env,
        platform_constraint = "@%s_config_platform//:platform_constraint" % name,
    )

def _tensorflow_local_config(name):
    remote_platform_configure(
        name = "%s_config_platform" % name,
        platform = "local",
        platform_exec_properties = {},
    )
    local_python_configure(
        name = "%s_config_python" % name,
        platform_constraint = "@%s_config_platform//:platform_constraint" % name,
    )

tensorflow_rbe_config = _tensorflow_rbe_config
tensorflow_rbe_win_config = _tensorflow_rbe_win_config
tensorflow_local_config = _tensorflow_local_config

# Streamlined platform configuration for the SIG Build containers.
# See //tensorflow/tools/tf_sig_build_dockerfiles
# These containers do not support ROCm and all have CUDA. We demand that the configuration
# provide all the env variables to remove hidden logic.
def sigbuild_tf_configs(name_container_map, env):
    for name, container in name_container_map.items():
        exec_properties = {
            "container-image": container,
            "Pool": "default",
        }

        remote_tensorrt_configure(
            name = "%s_config_tensorrt" % name,
            environ = env,
            exec_properties = exec_properties,
        )

        remote_platform_configure(
            name = "%s_config_platform" % name,
            platform = "linux",
            platform_exec_properties = exec_properties,
        )
