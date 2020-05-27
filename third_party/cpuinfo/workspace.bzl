"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-19b9316c71e4e45b170a664bf62ddefd7ac9feb5",
        sha256 = "e0a485c072de957668eb324c49d726dc0fd736cfb9436b334325f20d93085003",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/19b9316c71e4e45b170a664bf62ddefd7ac9feb5.zip",
            "https://github.com/pytorch/cpuinfo/archive/19b9316c71e4e45b170a664bf62ddefd7ac9feb5.zip",
        ],
        build_file = "//third_party/cpuinfo:BUILD.bazel",
    )
