"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
load("//third_party:repo.bzl", "tf_http_archive")

# The subset of LLVM targets that TensorFlow cares about.
_LLVM_TARGETS = [
    "AArch64",
    "AMDGPU",
    "ARM",
    "NVPTX",
    "PowerPC",
    "RISCV",
    "SystemZ",
    "X86",
    "SPIRV",
]

def llvm_setup(name):
    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(
        name = name,
        repo_mapping = {"@python_runtime": "@local_config_python"},
        targets = _LLVM_TARGETS,
    )

    tf_http_archive(
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

    tf_http_archive(
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )
