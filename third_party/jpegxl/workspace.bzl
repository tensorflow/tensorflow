"""Point to the JPEG XL repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "jpegxl",
        strip_prefix = "libjxl-0.11.1",
        sha256 = "1492dfef8dd6c3036446ac3b340005d92ab92f7d48ee3271b5dac1d36945d3d9",
        urls = tf_mirror_urls("https://github.com/libjxl/libjxl/archive/refs/tags/v0.11.1.tar.gz"),
        build_file = "//third_party/jpegxl:jpegxl.BUILD.bazel",
        patch_file = ["//third_party/jpegxl:external_deps.patch"],
    )
