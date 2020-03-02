"""Loads the pthreadpool library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pthreadpool",
        strip_prefix = "pthreadpool-4ea95bef8cdd942895f23f5cc09c778d10500551",
        sha256 = "100f675c099c74da46dea8da025f6f9b5e0307370f3dde506d11bd78b2b7d171",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/pthreadpool/archive/4ea95bef8cdd942895f23f5cc09c778d10500551.tar.gz",
            "https://github.com/Maratyszcza/pthreadpool/archive/4ea95bef8cdd942895f23f5cc09c778d10500551.tar.gz",
        ],
        build_file = "//third_party/pthreadpool:BUILD.bazel",
    )
