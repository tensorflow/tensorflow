"""Macro that creates external repositories for remote config."""

load("//tensorflow/tools/toolchains/remote_config:containers.bzl", "containers")
load("//third_party/gpus:rocm_configure.bzl", "remote_rocm_configure")
load("//third_party/py:python_configure.bzl", "local_python_configure", "remote_python_configure")
load("//third_party/remote_config:remote_platform_configure.bzl", "remote_platform_configure")

def _container_image_uri(container_name):
    container = containers[container_name]
    return "docker://%s/%s@%s" % (container["registry"], container["repository"], container["digest"])

def _tensorflow_rbe_config(name, os, rocm_version = None, cuda_version = None, cudnn_version = None):
    if cuda_version != None and rocm_version != None:
        fail("Specifying both cuda_version and rocm_version is not supported.")

    env = {}

    if cuda_version != None:
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
    elif rocm_version != None:
        # The rocm toolchain currently contains its own C++ toolchain definition,
        # so we do not fetch local_config_cc.
        env.update({
            "TF_NEED_ROCM": "1",
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

def _ml_build_rbe_config(container_image):
    exec_properties = {
        "container-image": container_image,
        "Pool": "default",
    }

    remote_platform_configure(
        name = "ml_build_config_platform",
        platform = "linux",
        platform_exec_properties = exec_properties,
    )

tensorflow_rbe_config = _tensorflow_rbe_config
tensorflow_rbe_win_config = _tensorflow_rbe_win_config
tensorflow_local_config = _tensorflow_local_config
ml_build_rbe_config = _ml_build_rbe_config

# TODO(b/369382309): Remove this once ml_build_rbe_config is used everywhere.
# Streamlined platform configuration for the SIG Build containers.
# See //tensorflow/tools/tf_sig_build_dockerfiles
# These containers do not support ROCm and all have CUDA.
def sigbuild_tf_configs(name_container_map):
    for name, container in name_container_map.items():
        exec_properties = {
            "container-image": container,
            "Pool": "default",
        }

        remote_platform_configure(
            name = "%s_config_platform" % name,
            platform = "linux",
            platform_exec_properties = exec_properties,
        )
