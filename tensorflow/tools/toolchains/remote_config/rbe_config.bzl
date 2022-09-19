"""Macro that creates external repositories for remote config."""

load("//third_party/py:python_configure.bzl", "local_python_configure", "remote_python_configure")
load("//third_party/gpus:cuda_configure.bzl", "remote_cuda_configure")
load("//third_party/nccl:nccl_configure.bzl", "remote_nccl_configure")
load("//third_party/gpus:rocm_configure.bzl", "remote_rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "remote_tensorrt_configure")
load("//tensorflow/tools/toolchains/remote_config:containers.bzl", "containers")
load("//third_party/remote_config:remote_platform_configure.bzl", "remote_platform_configure")

def _container_image_uri(container_name):
    container = containers[container_name]
    return "docker://%s/%s@%s" % (container["registry"], container["repository"], container["digest"])

def _tensorflow_rbe_config(name, compiler, python_versions, os, rocm_version = None, cuda_version = None, cudnn_version = None, tensorrt_version = None, tensorrt_install_path = None, cudnn_install_path = None, compiler_prefix = None, sysroot = None, python_install_path = "/usr"):
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
            "TF_NEED_CUDA": "1",
            "TF_CUDA_CLANG": "1" if compiler.endswith("clang") else "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_ENABLE_XLA": "1",
            "TF_CUDNN_VERSION": cudnn_version,
            "TF_CUDA_VERSION": cuda_version,
            "CUDNN_INSTALL_PATH": cudnn_install_path if cudnn_install_path != None else "/usr/lib/x86_64-linux-gnu",
            "TF_NEED_TENSORRT": "1",
            "TF_TENSORRT_VERSION": tensorrt_version,
            "TENSORRT_INSTALL_PATH": tensorrt_install_path if tensorrt_install_path != None else "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": compiler if not compiler.endswith("clang") else "",
            "GCC_HOST_COMPILER_PREFIX": compiler_prefix if compiler_prefix != None else "/usr/bin",
            "CLANG_CUDA_COMPILER_PATH": compiler if compiler.endswith("clang") else "",
            "TF_SYSROOT": sysroot if sysroot else "",
        })

        container_name = "cuda%s-cudnn%s-%s" % (cuda_version, cudnn_version, os)
        container_image = _container_image_uri(container_name)
        exec_properties = {
            "container-image": container_image,
            "Pool": "default",
        }

        remote_cuda_configure(
            name = "%s_config_cuda" % name,
            environ = env,
            exec_properties = exec_properties,
        )

        remote_nccl_configure(
            name = "%s_config_nccl" % name,
            environ = env,
            exec_properties = exec_properties,
        )

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
    elif python_versions != None:
        container_image = _container_image_uri(os)
        exec_properties = {
            "container-image": container_image,
            "Pool": "default",
        }

    else:
        fail("Neither cuda_version, rocm_version nor python_version specified.")

    remote_platform_configure(
        name = "%s_config_platform" % name,
        platform = "linux",
        platform_exec_properties = exec_properties,
    )
    for python_version in python_versions:
        env.update({
            "PYTHON_BIN_PATH": "%s/bin/python%s" % (python_install_path, python_version),
        })

        # For backwards compatibility do not add the python version to the name
        # if we only create a single python configuration.
        version = python_version if len(python_versions) > 1 else ""
        remote_python_configure(
            name = "%s_config_python%s" % (name, version),
            environ = env,
            exec_properties = exec_properties,
            platform_constraint = "@%s_config_platform//:platform_constraint" % name,
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
        environ = env,
        exec_properties = exec_properties,
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

        remote_cuda_configure(
            name = "%s_config_cuda" % name,
            environ = env,
            exec_properties = exec_properties,
        )

        remote_nccl_configure(
            name = "%s_config_nccl" % name,
            environ = env,
            exec_properties = exec_properties,
        )

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

        remote_python_configure(
            name = "%s_config_python" % name,
            environ = env,
            exec_properties = exec_properties,
            platform_constraint = "@%s_config_platform//:platform_constraint" % name,
        )
