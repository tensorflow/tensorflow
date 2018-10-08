"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "aws",
        urls = [
            "https://mirror.bazel.build/github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/1.3.15.tar.gz",
        ],
        sha256 = "b888d8ce5fc10254c3dd6c9020c7764dd53cf39cf011249d0b4deda895de1b7c",
        strip_prefix = "aws-sdk-cpp-1.3.15",
        build_file = "//third_party/aws:BUILD.bazel",
    )
