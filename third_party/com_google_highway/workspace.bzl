"""Point to the Highway repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "com_google_highway",
        strip_prefix = "highway-1.3.0",
        sha256 = "07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2",
        urls = tf_mirror_urls("https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz"),
        patch_file = ["//third_party/com_google_highway:build.patch"],
    )
