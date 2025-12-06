"""Point to the JPEGXL repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "jpegxl",
        strip_prefix = "libjxl-0.7.2",
        sha256 = "2564ddb891681fde7171da7c545c6cc9215f6b0f15093fb661fd2e9cb86bb640",
        urls = tf_mirror_urls("https://github.com/libjxl/libjxl/archive/refs/tags/v0.7.2.tar.gz"),
        build_file = "//third_party/jpegxl:jpegxl.BUILD.bazel",
    )
