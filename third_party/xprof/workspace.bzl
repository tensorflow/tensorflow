"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "d27bcd502a0843e463fc4eb7d3532d0d720ddd6af6e39942846f1aa769352625",
        strip_prefix = "xprof-c695e43eba127a74a67263775ab611bded7fba34",
        patch_file = ["//third_party/xprof:xprof.patch"],
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/c695e43eba127a74a67263775ab611bded7fba34.zip"),
        **kwargs
    )
