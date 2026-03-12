"""Loads the png library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "png",
        build_file = "//third_party/png:png.BUILD",
        patch_file = ["//third_party/png:png_fix_rpi.patch"],
        sha256 = "fecc95b46cf05e8e3fc8a414750e0ba5aad00d89e9fdf175e94ff041caf1a03a",
        strip_prefix = "libpng-1.6.43",
        system_build_file = "//third_party/systemlibs:png.BUILD",
        urls = tf_mirror_urls("https://github.com/glennrp/libpng/archive/v1.6.43.tar.gz"),
    )
