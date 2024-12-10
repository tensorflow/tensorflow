"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton/llvm_integration:series.bzl", "llvm_patch_list")
load("//third_party/triton/temporary:series.bzl", "temporary_patch_list")
load("//third_party/triton/xla_extensions:series.bzl", "extensions_files_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl702724623"
    TRITON_SHA256 = "7348c9fcc01f24d97daf71b9757b9065a36fedfe05a5fbe1ea79b603b89a65b9"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        patch_file = extensions_files_patch_list + llvm_patch_list + temporary_patch_list,
    )
