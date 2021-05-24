"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-3efc489b55385936531a06ff83425b719387ec63",
        sha256 = "b59586ce69bcf3efdbf3cf4803fadfeaae4948044e2b8d89cf912194cf28f233",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/dmlc/dlpack/archive/3efc489b55385936531a06ff83425b719387ec63.tar.gz",
            "https://github.com/dmlc/dlpack/archive/3efc489b55385936531a06ff83425b719387ec63.tar.gz",
        ],
        build_file = "//third_party/dlpack:BUILD.bazel",
    )
