"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton:llvm_integration/series.bzl", "llvm_patch_list")
load("//third_party/triton:temporary/series.bzl", "temporary_patch_list")
load("//third_party/triton:xla_extensions/series.bzl", "extensions_files_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl721789466"
    TRITON_SHA256 = "e89b5b362ad3ebdc05fb2db391ca27bd249262334a8721e0a037387d3f5f6525"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        patch_file = extensions_files_patch_list + llvm_patch_list + temporary_patch_list,
    )
