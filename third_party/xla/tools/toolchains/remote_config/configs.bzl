"""Configurations of RBE builds used with remote config."""

load("//tools/toolchains/remote_config:rbe_config.bzl", "sigbuild_tf_configs", "tensorflow_local_config", "tensorflow_rbe_config", "tensorflow_rbe_win_config")

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
        compiler = "/clang_rf2b94bd7eaa83d853dc7568fac87b1f8bf4ddec6/bin/clang",
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
        rocm_version = "5.0",  # Any version will do.
        os = "ubuntu18.04-manylinux2010-multipython",
        python_versions = ["2.7", "3.5", "3.6", "3.7", "3.8", "3.9"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-rocm",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        rocm_version = "5.3",  # Any version will do.
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.7", "3.8", "3.9", "3.10", "3.11"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2",
        compiler = "/clang_rf2b94bd7eaa83d853dc7568fac87b1f8bf4ddec6/bin/clang",
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

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda11.8-cudnn8.6-tensorrt8.4",
        compiler = "/clang_rf2b94bd7eaa83d853dc7568fac87b1f8bf4ddec6/bin/clang",
        cuda_version = "11.8",
        cudnn_version = "8.6",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "8.4",
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-cuda11.8-cudnn8.6-tensorrt8.4",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "11.8",
        cudnn_version = "8.6",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        tensorrt_install_path = "/usr",
        tensorrt_version = "8.4",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda12.1-cudnn8.9",
        compiler = "/usr/lib/llvm-17/bin/clang",
        cuda_version = "12.1",
        cudnn_version = "8.9",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-cuda12.1-cudnn8.9",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "12.1",
        cudnn_version = "8.9",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda12.2-cudnn8.9",
        compiler = "/usr/lib/llvm-17/bin/clang",
        cuda_version = "12.2",
        cudnn_version = "8.9",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-cuda12.2-cudnn8.9",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "12.2",
        cudnn_version = "8.9",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/usr/lib/llvm-17/bin/clang",
        cuda_version = "12.3",
        cudnn_version = "8.9",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-clang_manylinux2014-cuda12.3-cudnn9.1",
        compiler = "/usr/lib/llvm-17/bin/clang",
        cuda_version = "12.3",
        cudnn_version = "9.1",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu20.04-gcc9_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "12.3",
        cudnn_version = "8.9",
        os = "ubuntu20.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu22.04-clang_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/usr/lib/llvm-17/bin/clang",
        cuda_version = "12.3",
        cudnn_version = "8.9",
        os = "ubuntu22.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        sysroot = "/dt9",
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_config(
        name = "ubuntu22.04-gcc9_manylinux2014-cuda12.3-cudnn8.9",
        compiler = "/dt9/usr/bin/gcc",
        compiler_prefix = "/usr/bin",
        cuda_version = "12.3",
        cudnn_version = "8.9",
        os = "ubuntu22.04-manylinux2014-multipython",
        python_versions = ["3.9", "3.10", "3.11", "3.12"],
        python_install_path = "/usr/local",
    )

    tensorflow_rbe_win_config(
        name = "windows_py37",
        python_bin_path = "C:/Python37/python.exe",
    )

    # TF-Version-Specific SIG Build RBE Configs. The crosstool generated from these
    # configs are python-version-independent because they only care about the
    # tooling paths; the container mapping is useful only so that TF RBE users
    # may specify a specific Python version container. Yes, we could use the tag name instead,
    # but for vague security reasons we're obligated to use the pinned hash and update manually.
    # The name_container_map is helpfully auto-generated by a GitHub Action. You have to run it
    # manually. See go/tf-devinfra/docker#how-do-i-update-rbe-images

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.9": "docker://gcr.io/tensorflow-sigs/build@sha256:ce8e5b828a43ce2ea0a9d0a9d4f5d967a9bf79c0596b005a96c4ab91a8462347",
            "sigbuild-r2.9-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:d064667c4b904bb1c658a6be913fc41a1e5d822a5feb9cdac849973a050b9901",
            "sigbuild-r2.9-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:ce8e5b828a43ce2ea0a9d0a9d4f5d967a9bf79c0596b005a96c4ab91a8462347",
            "sigbuild-r2.9-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:5a3224e8d7f592b2f343e0c8fe9521605c7c02f51f0c5cc9f9652614d1961850",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.2",
            "TF_CUDNN_VERSION": "8.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.10": "docker://gcr.io/tensorflow-sigs/build@sha256:20d777f0200b7196349b9d25dec92ed4b34e966e8a8ab661d9b1b93c05d95c88",
            "sigbuild-r2.10-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:1091a4dc92c3227771ee748eb3f4eee1c32555f2e9805fcb341602b35e3da7a2",
            "sigbuild-r2.10-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:20d777f0200b7196349b9d25dec92ed4b34e966e8a8ab661d9b1b93c05d95c88",
            "sigbuild-r2.10-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:52c5bcfd3ce479c2f5148d7a9a119334148a33a3302b08e88e1045059dead62c",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.2",
            "TF_CUDNN_VERSION": "8.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-57469": "docker://gcr.io/tensorflow-sigs/build@sha256:771eb6cc8e4ba94b033f15a6b69d1d2eb52d28da6811f6e6a328ad814204679e",
            "sigbuild-57469-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:4341556586b640cd4b328959172e0a18767595e3446553c45353ef649d749388",
            "sigbuild-57469-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:771eb6cc8e4ba94b033f15a6b69d1d2eb52d28da6811f6e6a328ad814204679e",
            "sigbuild-57469-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:4d2241fea2a5ed629d9f7b68d9458bc0ce1f86651d02abcb596966c3cb92b492",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-15/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-15/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-15/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-15/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-15/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.2",
            "TF_CUDNN_VERSION": "8.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.11": "docker://gcr.io/tensorflow-sigs/build@sha256:19624dc8e664f4e00a85eee637711481ec00a22a9522a2575609f1ddce613615",
            "sigbuild-r2.11-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:bae2680dfb4457a9c6112aaf5f164dd677e4b14da0a1c6dabd81a573f8ec0d5d",
            "sigbuild-r2.11-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:19624dc8e664f4e00a85eee637711481ec00a22a9522a2575609f1ddce613615",
            "sigbuild-r2.11-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:5d55a95abee68d2e32ed5905708e0580b154939fd67e638e39bb4d2aa83d7ad6",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.2",
            "TF_CUDNN_VERSION": "8.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.12": "docker://gcr.io/tensorflow-sigs/build@sha256:bc94dcfc4b9e8e8abc91d67468d4af0345879c4d910cebc444d78402a7994237",
            "sigbuild-r2.12-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:8d12af5500f11ad2a4ff11cb1e967cfa559a158891eeae2e34d10cfacb87df22",
            "sigbuild-r2.12-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:bc94dcfc4b9e8e8abc91d67468d4af0345879c4d910cebc444d78402a7994237",
            "sigbuild-r2.12-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:5587846227d3ec090e9ad2ae559a3f4853aef7de013639ccca108fb910bf42a5",
            "sigbuild-r2.12-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:3a802d731a3455feb20aad39a0eb8f7f09be0ac4195f38dcd98154e8bb8bb6d4",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.8",
            "TF_CUDNN_VERSION": "8.6",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "8.4",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.12-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:bc94dcfc4b9e8e8abc91d67468d4af0345879c4d910cebc444d78402a7994237",
            "sigbuild-r2.12-clang-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:8d12af5500f11ad2a4ff11cb1e967cfa559a158891eeae2e34d10cfacb87df22",
            "sigbuild-r2.12-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:bc94dcfc4b9e8e8abc91d67468d4af0345879c4d910cebc444d78402a7994237",
            "sigbuild-r2.12-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:5587846227d3ec090e9ad2ae559a3f4853aef7de013639ccca108fb910bf42a5",
            "sigbuild-r2.12-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:3a802d731a3455feb20aad39a0eb8f7f09be0ac4195f38dcd98154e8bb8bb6d4",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-16/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-16/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-16/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-16/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-16/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.2",
            "TF_CUDNN_VERSION": "8.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.13": "docker://gcr.io/tensorflow-sigs/build@sha256:21131f082614f60207cb2242cd5150d5175a2a21e6789ad8fa32bd5eb7a1e5e0",
            "sigbuild-r2.13-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:95e55defa05f791e79beeaba5094341ce603cc00d8bdb5af5dc3496e4ed2f6e2",
            "sigbuild-r2.13-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:21131f082614f60207cb2242cd5150d5175a2a21e6789ad8fa32bd5eb7a1e5e0",
            "sigbuild-r2.13-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:d99a44bfe37c49fd1d08e94eca4de15682fe66017074c3feb695587f5bf5add9",
            "sigbuild-r2.13-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:684798c20fe171c932681cf54a4a80d27ed8fde6c0924ce96c9f7663eab1ef80",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.8",
            "TF_CUDNN_VERSION": "8.6",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "8.4",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.13-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:21131f082614f60207cb2242cd5150d5175a2a21e6789ad8fa32bd5eb7a1e5e0",
            "sigbuild-r2.13-clang-python3.8": "docker://gcr.io/tensorflow-sigs/build@sha256:95e55defa05f791e79beeaba5094341ce603cc00d8bdb5af5dc3496e4ed2f6e2",
            "sigbuild-r2.13-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:21131f082614f60207cb2242cd5150d5175a2a21e6789ad8fa32bd5eb7a1e5e0",
            "sigbuild-r2.13-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:d99a44bfe37c49fd1d08e94eca4de15682fe66017074c3feb695587f5bf5add9",
            "sigbuild-r2.13-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:684798c20fe171c932681cf54a4a80d27ed8fde6c0924ce96c9f7663eab1ef80",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-16/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-16/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-16/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-16/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-16/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "11.2",
            "TF_CUDNN_VERSION": "8.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "7.2",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.14": "docker://gcr.io/tensorflow-sigs/build@sha256:7c8ecb6482e26c4b4efce0ddaefe3fb3667b3b958c83fe8d3cc3763c6ed7a4d1",
            "sigbuild-r2.14-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:7c8ecb6482e26c4b4efce0ddaefe3fb3667b3b958c83fe8d3cc3763c6ed7a4d1",
            "sigbuild-r2.14-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:76d7505715334a34a0f96159e8c81350803ebef439726e5d50b7b6f5a7edc310",
            "sigbuild-r2.14-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:dbeb4c33bafaf83d7afcb2077440e83eec685ab2bc3dc520624ee8af69a34170",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.2",
            "TF_CUDNN_VERSION": "8.9",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "8.6",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.14-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:7c8ecb6482e26c4b4efce0ddaefe3fb3667b3b958c83fe8d3cc3763c6ed7a4d1",
            "sigbuild-r2.14-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:7c8ecb6482e26c4b4efce0ddaefe3fb3667b3b958c83fe8d3cc3763c6ed7a4d1",
            "sigbuild-r2.14-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:76d7505715334a34a0f96159e8c81350803ebef439726e5d50b7b6f5a7edc310",
            "sigbuild-r2.14-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:dbeb4c33bafaf83d7afcb2077440e83eec685ab2bc3dc520624ee8af69a34170",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-17/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-17/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.2",
            "TF_CUDNN_VERSION": "8.9",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "8.6",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.16": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:22d863e6fe3f98946015b9e1264b2eeb8e56e504535a6c1d5e564cae65ae5d37",
            "sigbuild-r2.16-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:da15288c8464153eadd35da720540a544b76aa9d78cceb42a6821b2f3e70a0fa",
            "sigbuild-r2.16-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:40fcd1d05c672672b599d9cb3784dcf379d6aa876f043b46c6ab18237d5d4e10",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.3",
            "TF_CUDNN_VERSION": "8.9",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "8.6",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.16-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:22d863e6fe3f98946015b9e1264b2eeb8e56e504535a6c1d5e564cae65ae5d37",
            "sigbuild-r2.16-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:da15288c8464153eadd35da720540a544b76aa9d78cceb42a6821b2f3e70a0fa",
            "sigbuild-r2.16-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:842a5ba84d3658c5bf1f8a31e16284f7becc35409da0dfd71816afa3cd28d728",
            "sigbuild-r2.16-clang-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:40fcd1d05c672672b599d9cb3784dcf379d6aa876f043b46c6ab18237d5d4e10",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-17/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-17/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.3",
            "TF_CUDNN_VERSION": "8.9",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_NEED_TENSORRT": "1",
            "TF_SYSROOT": "/dt9",
            "TF_TENSORRT_VERSION": "8.6",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.17": "docker://gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545",
            "sigbuild-r2.17-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:8ca6b205b54f18d26a053cfe606145b8b11cc99cf83fc970a936ce327913c3c3",
            "sigbuild-r2.17-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:5cfd081a337548165a800546f2365a38245e38e7a97052b1a21830bf66b2356d",
            "sigbuild-r2.17-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545",
            "sigbuild-r2.17-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:933c9f4bf65c92780863e00bd2132c6cfd41dbd624736c1af0dd2a5a056db6b8",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt9/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/dt9/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": "/dt9/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt9/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt9/usr/bin/gcc",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TF_CUDA_CLANG": "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.3",
            "TF_CUDNN_VERSION": "8.9",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_SYSROOT": "/dt9",
            "TF_NEED_TENSORRT": "1",
            "TF_TENSORRT_VERSION": "8.6",
        },
    )

    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.17-clang": "docker://gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545",
            "sigbuild-r2.17-clang-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:8ca6b205b54f18d26a053cfe606145b8b11cc99cf83fc970a936ce327913c3c3",
            "sigbuild-r2.17-clang-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:5cfd081a337548165a800546f2365a38245e38e7a97052b1a21830bf66b2356d",
            "sigbuild-r2.17-clang-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:dddcaf30321e9007103dce75c51b83fea3c06de462fcf41e7c6ae93f37fc3545",
            "sigbuild-r2.17-clang-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:933c9f4bf65c92780863e00bd2132c6cfd41dbd624736c1af0dd2a5a056db6b8",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-17/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-17/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.3",
            "TF_CUDNN_VERSION": "8.9",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_SYSROOT": "/dt9",
            "TF_NEED_TENSORRT": "1",
            "TF_TENSORRT_VERSION": "8.6",
        },
    )

    # TODO(b/338885148): Remove this temporary RBE config once the TF standard config is on cuDNN 9
    sigbuild_tf_configs(
        name_container_map = {
            "sigbuild-r2.17-clang-cudnn9": "docker://gcr.io/tensorflow-sigs/build@sha256:52420ff74ce5179fed76d72ac37dafeae3d111a3e7862950ce186c841876e254",
            "sigbuild-r2.17-clang-cudnn9-python3.9": "docker://gcr.io/tensorflow-sigs/build@sha256:52420ff74ce5179fed76d72ac37dafeae3d111a3e7862950ce186c841876e254",
            "sigbuild-r2.17-clang-cudnn9-python3.10": "docker://gcr.io/tensorflow-sigs/build@sha256:c6e96290045ea5ec7c61ef2d3e07335089a3d778814f3859914f460e91ae2f79",
            "sigbuild-r2.17-clang-cudnn9-python3.11": "docker://gcr.io/tensorflow-sigs/build@sha256:3a5581f35aa2daf6a168d023802e1e3cac1169755a02fb5498ff9756ad3598b5",
            "sigbuild-r2.17-clang-cudnn9-python3.12": "docker://gcr.io/tensorflow-sigs/build@sha256:70a1414721826d8c899e2bc508ea7265828629af949cf1f6753b5ee12a9559b2",
        },
        # Unclear why LIBC is set to 2.19 here, and yet manylinux2010 is 2.12
        # and manylinux2014 is 2.17.
        env = {
            "ABI_LIBC_VERSION": "glibc_2.19",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "BAZEL_HOST_SYSTEM": "i686-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "k8",
            "BAZEL_TARGET_LIBC": "glibc_2.19",
            "BAZEL_TARGET_SYSTEM": "x86_64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-17/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_gnu_x86",
            "CLEAR_CACHE": "1",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "CLANG_CUDA_COMPILER_PATH": "/usr/lib/llvm-17/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-17/bin/clang",
            "PYTHON_BIN_PATH": "/usr/bin/python3",
            "TF_CUDA_CLANG": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.5,6.0",
            "TF_CUDA_VERSION": "12.3",
            "TF_CUDNN_VERSION": "9.1",
            "TF_ENABLE_XLA": "1",
            "TF_NEED_CUDA": "1",
            "TF_SYSROOT": "/dt9",
            "TF_NEED_TENSORRT": "1",
            "TF_TENSORRT_VERSION": "10.0",
        },
    )
