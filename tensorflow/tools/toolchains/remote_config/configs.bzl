"""Configurations of RBE builds used with remote config."""

load("//tensorflow/tools/toolchains/remote_config:rbe_config.bzl", "sigbuild_tf_configs", "tensorflow_local_config", "tensorflow_rbe_config", "tensorflow_rbe_win_config")

def initialize_rbe_configs():
    tensorflow_local_config(
        name = "local_execution",
    )

    tensorflow_rbe_config(
        name = "ubuntu16.04-manylinux2010-py3",
        os = "ubuntu16.04-manylinux2010",
        python_versions = ["3"],
        compiler = "",
    )

    tensorflow_rbe_config(
        name = "ubuntu16.04-py3-gcc7_manylinux2010-cuda10.0-cudnn7-tensorrt5.1",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "10.0",
        cudnn_version = "7",
        os = "ubuntu16.04-manylinux2010",
        python_versions = ["3"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "5.1",
    )

    tensorflow_rbe_config(
        name = "ubuntu16.04-gcc7_manylinux2010-cuda10.1-cudnn7-tensorrt6.0",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "10.1",
        cudnn_version = "7",
        os = "ubuntu16.04-manylinux2010-multipython",
        python_versions = ["2.7", "3.5", "3.6", "3.7", "3.8"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "6.0",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda10.1-cudnn7-tensorrt6.0",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "10.1",
        cudnn_version = "7",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["2.7", "3.5", "3.6", "3.7", "3.8"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "6.0",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda10.2-cudnn7-tensorrt6.0",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "10.2",
        cudnn_version = "7",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "6.0",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda11.0-cudnn8-tensorrt7.1",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.0",
        cudnn_version = "8",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["2.7", "3.5", "3.6", "3.7", "3.8"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.1",
        python_install_path = "/usr/local",
    )

    # TODO(klimek): Delete this once all users are migrated to a python-version
    # independent configuration. In the future, use
    # "ubuntu16.04-gcc7_manylinux2010-cuda10.1-cudnn7-tensorrt6.0" instead.
    tensorflow_rbe_config(
        name = "ubuntu16.04-py3-gcc7_manylinux2010-cuda10.1-cudnn7-tensorrt6.0",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "10.1",
        cudnn_version = "7",
        os = "ubuntu16.04-manylinux2010",
        python_versions = ["3"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "6.0",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-clang_manylinux2010-cuda11.0-cudnn8-tensorrt7.1",
        compiler = "/clang_r969a51ff363263a3b5f2df55eba6b4d392bf30c0/bin/clang",
        cuda_version = "11.0",
        cudnn_version = "8",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["2.7", "3.5", "3.6", "3.7", "3.8"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.1",
        sysroot = "/dt7",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-rocm",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        rocm_version = "3.5",  # Any version will do.
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["2.7", "3.5", "3.6", "3.7", "3.8"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-clang_manylinux2010-cuda11.1-cudnn8-tensorrt7.2",
        compiler = "/clang_r969a51ff363263a3b5f2df55eba6b4d392bf30c0/bin/clang",
        cuda_version = "11.1",
        cudnn_version = "8",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        sysroot = "/dt7",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda11.1-cudnn8-tensorrt7.2",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.1",
        cudnn_version = "8",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-clang_manylinux2010-cuda11.2-cudnn8.1-tensorrt7.2",
        compiler = "/clang_r969a51ff363263a3b5f2df55eba6b4d392bf30c0/bin/clang",
        cuda_version = "11.2",
        cudnn_version = "8.1",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        sysroot = "/dt7",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda11.2-cudnn8.1-tensorrt7.2",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.2",
        cudnn_version = "8.1",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-clang_manylinux2010-cuda11.4-cudnn8.0.5-tensorrt7.2",
        compiler = "/clang_r969a51ff363263a3b5f2df55eba6b4d392bf30c0/bin/clang",
        cuda_version = "11.4",
        cudnn_version = "8.0.5",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        sysroot = "/dt7",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda11.4-cudnn8.0.5-tensorrt7.2",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.4",
        cudnn_version = "8.0.5",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-clang_manylinux2010-cuda11.4-cudnn8.2-tensorrt7.2",
        compiler = "/clang_r969a51ff363263a3b5f2df55eba6b4d392bf30c0/bin/clang",
        cuda_version = "11.4",
        cudnn_version = "8.2",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        sysroot = "/dt7",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu18.04-gcc7_manylinux2010-cuda11.4-cudnn8.2-tensorrt7.2",
        compiler = "/dt7/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.4",
        cudnn_version = "8.2",
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2",
        compiler = "/clang11/bin/clang",
        cuda_version = "11.2",
        cudnn_version = "8.1",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.2",
        cudnn_version = "8.1",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "7.2",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_win_config(
        name = "windows_py37",
        python_bin_path = "C:/Python37/python.exe",
    )

    # Experimental SIG Build RBE Config. The crosstool generated from this
    # config is python-version-independent because it only cares about the
    # tooling paths; the container mapping is useful only so that TF RBE users
    # may specify a specific Python version container.
    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.9": "docker://gcr.io/tensorflow-sigs/build@sha256:fb84093359afd55b2cf3cebfe64e8281b0f67a44405ed338b2832ae910322c9c",
            "sigbuild-r2.9-python3.7": "docker://gcr.io/tensorflow-sigs/build@sha256:84b7f306e2d89d77966a6f457f4ffd7ccadefa5022086e285260dbf8c4065bd9",
            "sigbuild-r2.9-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:cf4155e40bbc2e68669e1e2e1f990e8cd5dc2ab9bc8c67e9679e16f6cd9712f5",
            "sigbuild-r2.9-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:fb84093359afd55b2cf3cebfe64e8281b0f67a44405ed338b2832ae910322c9c",
            "sigbuild-r2.9-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:382391982e7882ca1f6d2d7e39ca35fec6d47a11110ef83beba314ef802c5b67",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt7/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt7/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt7/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt7/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt7/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "8.1",
            "TF_CUDNN_VERSION": "11.2",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt7",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    # Double-experimental config for manylinux2014 PR:
    # https://github.com/tensorflow/build/pull/57
    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-57": "docker://gcr.io/tensorflow-sigs/build@sha256:0a2e12ca7ab8536a31f1854f72510986a6792413c9d5815535486571664402d8",
            "sigbuild-57-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:0a2e12ca7ab8536a31f1854f72510986a6792413c9d5815535486571664402d8",
        },
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt8/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt8/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt8/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt8/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt8/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "8.1",
            "TF_CUDNN_VERSION": "11.2",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt8",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )
