"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton:llvm_integration/series.bzl", "llvm_patch_list")
load("//third_party/triton:temporary/series.bzl", "temporary_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "triton_integrate_branch-1.11"
    TRITON_SHA256 = "1125fd9e344de2cb4041e4a9ec2cf02c307082833e421d87f91ffcf9983f9a90"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-" + TRITON_COMMIT,
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{}.tar.gz".format(TRITON_COMMIT)),
        patch_file = llvm_patch_list + temporary_patch_list,
    )
