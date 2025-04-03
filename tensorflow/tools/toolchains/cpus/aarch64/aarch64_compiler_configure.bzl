"""Configurations of AARCH64 builds used with Docker container."""

load("@local_xla//third_party/remote_config:remote_platform_configure.bzl", "remote_platform_configure")
load("//tensorflow/tools/toolchains:cpus/aarch64/aarch64.bzl", "remote_aarch64_configure")

def ml2014_tf_aarch64_configs(name_container_map, env):
    for name, container in name_container_map.items():
        exec_properties = {
            "container-image": container,
            "Pool": "default",
        }

        remote_aarch64_configure(
            name = "%s_config_aarch64" % name,
            environ = env,
            exec_properties = exec_properties,
        )

        remote_platform_configure(
            name = "%s_config_aarch64_platform" % name,
            platform = "linux",
            platform_exec_properties = exec_properties,
        )

def aarch64_compiler_configure():
    ml2014_tf_aarch64_configs(
        name_container_map = {
            "ml2014_aarch64": "docker://localhost/tensorflow-build-aarch64",
            "ml2014_aarch64-python3.9": "docker://localhost/tensorflow-build-aarch64:latest-python3.9",
            "ml2014_aarch64-python3.10": "docker://localhost/tensorflow-build-aarch64:latest-python3.10",
            "ml2014_aarch64-python3.11": "docker://localhost/tensorflow-build-aarch64:latest-python3.11",
        },
        env = {
            "ABI_LIBC_VERSION": "glibc_2.17",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/dt10/usr/bin/gcc",
            "BAZEL_HOST_SYSTEM": "aarch64-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "generic",
            "BAZEL_TARGET_LIBC": "glibc_2.17",
            "BAZEL_TARGET_SYSTEM": "aarch64-unknown-linux-gnu",
            "CC": "/dt10/usr/bin/gcc",
            "CC_TOOLCHAIN_NAME": "linux_gnu_aarch64",
            "CLEAR_CACHE": "1",
            "GCC_HOST_COMPILER_PATH": "/dt10/usr/bin/gcc",
            "GCC_HOST_COMPILER_PREFIX": "/usr/bin",
            "HOST_CXX_COMPILER": "/dt10/usr/bin/gcc",
            "HOST_C_COMPILER": "/dt10/usr/bin/gcc",
            "TF_ENABLE_XLA": "1",
            "TF_SYSROOT": "/dt10",
        },
    )

    ml2014_tf_aarch64_configs(
        name_container_map = {
            "ml2014_clang_aarch64": "docker://localhost/tensorflow-build-aarch64",
            "ml2014_clang_aarch64-python3.9": "docker://localhost/tensorflow-build-aarch64:latest-python3.9",
            "ml2014_clang_aarch64-python3.10": "docker://localhost/tensorflow-build-aarch64:latest-python3.10",
            "ml2014_clang_aarch64-python3.11": "docker://localhost/tensorflow-build-aarch64:latest-python3.11",
            "ml2014_clang_aarch64-python3.12": "docker://localhost/tensorflow-build-aarch64:latest-python3.12",
        },
        env = {
            "ABI_LIBC_VERSION": "glibc_2.17",
            "ABI_VERSION": "gcc",
            "BAZEL_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "BAZEL_HOST_SYSTEM": "aarch64-unknown-linux-gnu",
            "BAZEL_TARGET_CPU": "generic",
            "BAZEL_TARGET_LIBC": "glibc_2.17",
            "BAZEL_TARGET_SYSTEM": "aarch64-unknown-linux-gnu",
            "CC": "/usr/lib/llvm-18/bin/clang",
            "CC_TOOLCHAIN_NAME": "linux_llvm_aarch64",
            "CLEAR_CACHE": "1",
            "CLANG_COMPILER_PATH": "/usr/lib/llvm-18/bin/clang",
            "HOST_CXX_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "HOST_C_COMPILER": "/usr/lib/llvm-18/bin/clang",
            "TF_ENABLE_XLA": "1",
            "TF_SYSROOT": "/dt10",
        },
    )
