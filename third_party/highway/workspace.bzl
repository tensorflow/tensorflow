"""Point to the Highway repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "highway",
        strip_prefix = "highway-0.18.1",
        sha256 = "07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2",
        urls = tf_mirror_urls("https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz"),
        build_file = "//third_party/highway:highway.BUILD.bazel",
    )
