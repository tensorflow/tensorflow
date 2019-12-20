"""Loads the pthreadpool library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pthreadpool",
        strip_prefix = "pthreadpool-7ad026703b3109907ad124025918da15cfd3f100",
        sha256 = "96eb4256fc438b7b8cab40541d383efaf546fae7bad380c24ea601c326c5f685",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/pthreadpool/archive/7ad026703b3109907ad124025918da15cfd3f100.tar.gz",
            "https://github.com/Maratyszcza/pthreadpool/archive/7ad026703b3109907ad124025918da15cfd3f100.tar.gz",
        ],
        build_file = "//third_party/pthreadpool:BUILD.bazel",
    )
