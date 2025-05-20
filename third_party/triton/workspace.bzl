"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton:llvm_integration/series.bzl", "llvm_patch_list")
load("//third_party/triton:temporary/series.bzl", "temporary_patch_list")
load("//third_party/triton:xla_extensions/series.bzl", "extensions_files_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "triton_integrate_branch-1.6"
    TRITON_SHA256 = "323e1f04c5eaade72cf69fbe719487681d2a5108a86f82d0ed4c7ea61dfcea27"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-" + TRITON_COMMIT,
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{}.tar.gz".format(TRITON_COMMIT)),
        patch_file = extensions_files_patch_list + llvm_patch_list + temporary_patch_list,
    )
