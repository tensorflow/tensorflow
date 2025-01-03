""" Downloads clang and configures the crosstool using bazel's autoconf."""

load("@bazel_tools//tools/cpp:cc_configure.bzl", "cc_autoconf_impl")
load(":download_clang.bzl", "download_llvm_clang")

_TF_DOWNLOAD_CLANG = "TF_DOWNLOAD_CLANG"
_TF_NEED_CUDA = "TF_NEED_CUDA"

_DUMMY_BUILD_CONTENT = """
filegroup(
  name = "clang",
  visibility = ["//visibility:public"],
)
"""

_DOWNLOADED_ARCHIVE_BUILD_CONTENT = """
filegroup(
    name = "clang",
    srcs = [
        "bin/clang",
    ],
    visibility = ["//visibility:public"],
)
"""

def _cc_clang_autoconf(repo_ctx):
    if repo_ctx.os.environ.get(_TF_DOWNLOAD_CLANG) != "1":
        repo_ctx.file("BUILD", _DUMMY_BUILD_CONTENT)
        return

    download_llvm_clang(repo_ctx)
    repo_ctx.file("BUILD", _DOWNLOADED_ARCHIVE_BUILD_CONTENT)

    if repo_ctx.os.environ.get(_TF_NEED_CUDA) == "1":
        # Clang is handled separately for CUDA configs.
        # See cuda_configure.bzl for more details.
        return

    overridden_tools = {"gcc": "bin/clang"}
    cc_autoconf_impl(repo_ctx, overridden_tools)

cc_download_clang_toolchain = repository_rule(
    environ = [
        _TF_DOWNLOAD_CLANG,
        _TF_NEED_CUDA,
    ],
    implementation = _cc_clang_autoconf,
)
