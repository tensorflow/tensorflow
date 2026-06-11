"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "51fc9986e8b4a7d2f6333d914454e333fc955076de06d06826ae6babf02e6310",
        strip_prefix = "xprof-c2e053062acdcdf2119eb1ab89d19b4ac6ebe9fc",
        patch_file = ["//third_party/xprof:xprof.patch"],
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/c2e053062acdcdf2119eb1ab89d19b4ac6ebe9fc.zip"),
        **kwargs
    )
