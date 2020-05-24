"""Loads the FP16 library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "FP16",
        strip_prefix = "FP16-3c54eacb74f6f5e39077300c5564156c424d77ba",
        sha256 = "0d56bb92f649ec294dbccb13e04865e3c82933b6f6735d1d7145de45da700156",
        urls = [
            "https://mirror.bazel.build/github.com/Maratyszcza/FP16/archive/3c54eacb74f6f5e39077300c5564156c424d77ba.zip",
            "https://github.com/Maratyszcza/FP16/archive/3c54eacb74f6f5e39077300c5564156c424d77ba.zip",
        ],
        build_file = "//third_party/FP16:BUILD.bazel",
    )
