"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ortools_archive",
        urls = [
            "https://mirror.bazel.build/github.com/google/or-tools/archive/v6.7.2.tar.gz",
            "https://github.com/google/or-tools/archive/v6.7.2.tar.gz",
        ],
        sha256 = "d025a95f78b5fc5eaa4da5f395f23d11c23cf7dbd5069f1f627f002de87b86b9",
        strip_prefix = "or-tools-6.7.2/src",
        build_file = "//third_party/ortools:BUILD.bazel",
    )
