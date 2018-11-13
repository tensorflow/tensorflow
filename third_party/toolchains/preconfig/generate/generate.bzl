load(
    "@bazel_toolchains//rules:docker_config.bzl",
    "docker_toolchain_autoconfig",
)

def _tensorflow_rbe_config(name, cuda_version, cudnn_version, python_version, compiler):
    docker_toolchain_autoconfig(
        name = name,
        base = "@cuda%s-cudnn%s-ubuntu14.04//image" % (cuda_version, cudnn_version),
        bazel_version = "0.16.1",
        config_repos = [
            "local_config_cuda",
            "local_config_python",
            "local_config_nccl",
        ],
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
            "PYTHON_BIN_PATH": "/usr/bin/python%s" % python_version,
            "TF_NEED_CUDA": "1",
            "TF_CUDA_CLANG": "1" if compiler == "clang" else "0",
            "CLEAR_CACHE": "1",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.0",
            "TF_ENABLE_XLA": "1",
            "TF_CUDNN_VERSION": cudnn_version,
            "TF_CUDA_VERSION": cuda_version,
            "NCCL_INSTALL_PATH": "/usr/lib",
            "NCCL_HDR_PATH": "/usr/include",
            "TF_NCCL_VERSION": "2",
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
        },
        # TODO(klimek): We should use the sources that we currently work on, not
        # just the latest snapshot of tensorflow that is checked in.
        git_repo = "https://github.com/tensorflow/tensorflow",
        tags = ["manual"],
        incompatible_changes_off = True,
    )

tensorflow_rbe_config = _tensorflow_rbe_config
