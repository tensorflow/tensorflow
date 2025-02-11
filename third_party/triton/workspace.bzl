"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton/llvm_integration:series.bzl", "llvm_patch_list")
load("//third_party/triton/temporary:series.bzl", "temporary_patch_list")
load("//third_party/triton/xla_extensions:series.bzl", "extensions_files_patch_list")

def repo():
    """Imports Triton."""

<<<<<<< HEAD
    TRITON_COMMIT = "cl715271136"
    TRITON_SHA256 = "f08445eac5df52173b50aebfb0a811b295287e2657f5ef73e778b3feface8d68"
=======
    TRITON_COMMIT = "cl722997049"
    TRITON_SHA256 = "3c5de0e06947cc8cd6b6b06d0c28b3234f6ff72a4ae8f63c80e6d276413e5d7e"
>>>>>>> google/master
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        patch_file = extensions_files_patch_list + llvm_patch_list + temporary_patch_list,
    )
