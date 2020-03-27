"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-0cc563acb9baac39f2c1349bc42098c4a1da59e3",
        sha256 = "80625d0b69a3d69b70c2236f30db2c542d0922ccf9bb51a61bc39c49fac91a35",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/0cc563acb9baac39f2c1349bc42098c4a1da59e3.tar.gz",
            "https://github.com/pytorch/cpuinfo/archive/0cc563acb9baac39f2c1349bc42098c4a1da59e3.tar.gz",
        ],
        build_file = "//third_party/cpuinfo:BUILD.bazel",
    )
