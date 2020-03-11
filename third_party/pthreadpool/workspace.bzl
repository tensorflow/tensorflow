"""Loads the pthreadpool library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pthreadpool",
        strip_prefix = "pthreadpool-ebd50d0cfa3664d454ffdf246fcd228c3b370a11",
        sha256 = "ca4fc774cf2339cb739bba827de8ed4ccbd450c4608e05329e974153448aaf56",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/pthreadpool/archive/ebd50d0cfa3664d454ffdf246fcd228c3b370a11.tar.gz",
            "https://github.com/Maratyszcza/pthreadpool/archive/ebd50d0cfa3664d454ffdf246fcd228c3b370a11.tar.gz",
        ],
        build_file = "//third_party/pthreadpool:BUILD.bazel",
    )
