load(
    "@bazel_toolchains//rules:docker_config.bzl",
    "docker_toolchain_autoconfig",
)

def _tensorflow_rbe_config(name, compiler, python_version, cuda_version = None, cudnn_version = None, tensorrt_version = None):
    base = "@ubuntu16.04//image"
    config_repos = [
        "local_config_python",
        "local_config_cc",
    ]
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
        "CLEAR_CACHE": "1",
        "HOST_CXX_COMPILER": compiler,
        "HOST_C_COMPILER": compiler,
    }

    if cuda_version != None:
        base = "@cuda%s-cudnn%s-ubuntu14.04//image" % (cuda_version, cudnn_version)

        # The cuda toolchain currently contains its own C++ toolchain definition,
        # so we do not fetch local_config_cc.
        config_repos = [
            "local_config_python",
            "local_config_cuda",
            "local_config_tensorrt",
        ]
        env.update({
            "TF_NEED_CUDA": "1",
            "TF_CUDA_CLANG": "1" if compiler == "clang" else "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.0,6.0",
            "TF_ENABLE_XLA": "1",
            "TF_CUDNN_VERSION": cudnn_version,
            "TF_CUDA_VERSION": cuda_version,
            "CUDNN_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "TF_NEED_TENSORRT": "1",
            "TF_TENSORRT_VERSION": tensorrt_version,
            "TENSORRT_INSTALL_PATH": "/usr/lib/x86_64-linux-gnu",
            "GCC_HOST_COMPILER_PATH": compiler if compiler != "clang" else "",
        })

    docker_toolchain_autoconfig(
        name = name,
        base = base,
        bazel_version = "0.21.0",
        config_repos = config_repos,
        env = env,
        mount_project = "$(mount_project)",
        tags = ["manual"],
        incompatible_changes_off = True,
    )

tensorflow_rbe_config = _tensorflow_rbe_config
