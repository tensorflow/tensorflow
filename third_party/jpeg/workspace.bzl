"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "libjpeg_turbo",
        urls = tf_mirror_urls("https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/2.1.4.tar.gz"),
        sha256 = "a78b05c0d8427a90eb5b4eb08af25309770c8379592bb0b8a863373128e6143f",
        strip_prefix = "libjpeg-turbo-2.1.4",
        build_file = "//third_party/jpeg:jpeg.BUILD",
        system_build_file = "//third_party/jpeg:BUILD.system",
    )
