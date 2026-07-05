"""Provides the repository macro to import raft."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def tensorflow_repo():
    """Imports raft."""

    RAFT_VERSION = "25.08.00"
    RAFT_SHA256 = "032dce57b297e121352a1556bd9021410be30fcf319e158592f615e1990b2e58"

    tf_http_archive(
        name = "raft",
        sha256 = RAFT_SHA256,
        strip_prefix = "raft-{version}".format(version = RAFT_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/raft/archive/refs/tags/v{version}.tar.gz".format(version = RAFT_VERSION)),
        build_file = "//third_party/raft:raft.BUILD",
        patch_file = [
            "//third_party/raft:cudart_utils.hpp.patch",
            "//third_party/raft:vectorized.cuh.patch",
            "//third_party/raft:clang_cuda_intrinsics.h.patch",
            "//third_party/raft:pr-2807.patch",
            "//third_party/raft:logger_macros.hpp.patch",
            "//third_party/raft:select_k_runner.hpp.patch",
            "//third_party/raft:select_k_runner.cu.cc.patch",
            "//third_party/raft:select_k_smoke_test.cu.cc.patch",
        ],
    )

def xla_repo():
    """Imports raft."""

    RAFT_VERSION = "26.02.00"
    RAFT_SHA256 = "b4c005c70d9e1281a65dc30dc39051d1e4e713130da1fa0b76df832064674b59"

    tf_http_archive(
        name = "raft",
        sha256 = RAFT_SHA256,
        strip_prefix = "raft-{version}".format(version = RAFT_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/raft/archive/refs/tags/v{version}.tar.gz".format(version = RAFT_VERSION)),
        build_file = "//third_party/raft:raft.xla.BUILD",
        patch_file = [
            "//third_party/raft:clang_cuda_intrinsics.h.patch",
            "//third_party/raft:logger_macros.hpp.patch",
            "//third_party/raft:select_k_runner.hpp.patch",
            "//third_party/raft:select_k_runner.cu.cc.patch",
            "//third_party/raft:select_k_smoke_test.cu.cc.patch",
        ],
    )
