"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "libjpeg_turbo",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.tar.gz",
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.tar.gz",
        ],
        sha256 = "b3090cd37b5a8b3e4dbd30a1311b3989a894e5d3c668f14cbc6739d77c9402b7",
        strip_prefix = "libjpeg-turbo-2.0.5",
        build_file = "//third_party/jpeg:BUILD.bazel",
        system_build_file = "//third_party/jpeg:BUILD.system",
    )
