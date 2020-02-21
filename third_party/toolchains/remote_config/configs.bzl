"""Configurations of RBE builds used with remote config."""

load("//third_party/toolchains/remote_config:rbe_config.bzl", "tensorflow_rbe_config")

def initialize_rbe_configs():
    tensorflow_rbe_config(
        name = "ubuntu16.04-py3-gcc7_manylinux2010-cuda10.0-cudnn7-tensorrt5.1",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "10.0",
        cudnn_version = "7",
        os = "ubuntu16.04-manylinux2010",
        python_version = "3",
        tensorrt_install_path = "/usr",
        tensorrt_version = "5.1",
    )

    tensorflow_rbe_config(
        name = "ubuntu16.04-py3_opt-gcc5-rocm",
        compiler = "gcc",
        os = "ubuntu16.04",
        python_version = "3",
        rocm_version = "2.5",  # Any version will do.
    )
