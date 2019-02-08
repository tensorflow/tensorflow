"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "jpeg",
        urls = [
            "https://mirror.bazel.build/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.0.tar.gz",
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.0.tar.gz",
        ],
        sha256 = "f892fff427ab3adffc289363eac26d197ce3ccacefe5f5822377348a8166069b",
        strip_prefix = "libjpeg-turbo-2.0.0",
        build_file = "//third_party/jpeg:BUILD.bazel",
        system_build_file = "//third_party/jpeg:BUILD.system",
    )
