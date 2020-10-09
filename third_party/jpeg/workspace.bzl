"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "libjpeg_turbo",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.4.tar.gz",
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.4.tar.gz",
        ],
        sha256 = "7777c3c19762940cff42b3ba4d7cd5c52d1671b39a79532050c85efb99079064",
        strip_prefix = "libjpeg-turbo-2.0.4",
        build_file = "//third_party/jpeg:BUILD.bazel",
        system_build_file = "//third_party/jpeg:BUILD.system",
    )
