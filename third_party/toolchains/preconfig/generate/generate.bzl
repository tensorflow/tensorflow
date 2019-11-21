load(
    "@bazel_toolchains//rules:docker_config.bzl",
    "docker_toolchain_autoconfig",
)

def _tensorflow_rbe_config(name, compiler, python_version, os, rocm_version = None, cuda_version = None, cudnn_version = None, tensorrt_version = None, tensorrt_install_path = None, cudnn_install_path = None, compiler_prefix = None, build_bazel_src = False, sysroot = None):
    base = "@%s//image" % os
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

    if cuda_version != None and rocm_version != None:
        fail("Specifying both cuda_version and rocm_version is not supported.")

    if cuda_version != None:
        base = "@cuda%s-cudnn%s-%s//image" % (cuda_version, cudnn_version, os)

        # The cuda toolchain currently contains its own C++ toolchain definition,
        # so we do not fetch local_config_cc.
        config_repos = [
            "local_config_python",
            "local_config_cuda",
            "local_config_tensorrt",
        ]
        env.update({
            "TF_NEED_CUDA": "1",
            "TF_CUDA_CLANG": "1" if compiler.endswith("clang") else "0",
            "TF_CUDA_COMPUTE_CAPABILITIES": "3.0,6.0",
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

    if rocm_version != None:
        base = "@rocm-%s//image" % (os)

        # The rocm toolchain currently contains its own C++ toolchain definition,
        # so we do not fetch local_config_cc.
        config_repos = [
            "local_config_python",
            "local_config_rocm",
        ]
        env.update({
            "TF_NEED_ROCM": "1",
            "TF_ENABLE_XLA": "0",
        })

    docker_toolchain_autoconfig(
        name = name,
        base = base,
        bazel_version = "0.29.1",
        build_bazel_src = build_bazel_src,
        config_repos = config_repos,
        env = env,
        mount_project = "$(mount_project)",
        tags = ["manual"],
    )

tensorflow_rbe_config = _tensorflow_rbe_config
