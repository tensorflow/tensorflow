"""Loads the psimd library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "psimd",
        strip_prefix = "psimd-85427dd4c8521cc037a1ffa6fcd25c55fafc8a00",
        sha256 = "db23c2bc4a58d6f40c181797e43103300edac7cf9d286ca81590543f66ab95d2",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/psimd/archive/85427dd4c8521cc037a1ffa6fcd25c55fafc8a00.zip",
            "https://github.com/Maratyszcza/psimd/archive/85427dd4c8521cc037a1ffa6fcd25c55fafc8a00.zip",
        ],
        build_file = "//third_party/psimd:BUILD.bazel",
    )
