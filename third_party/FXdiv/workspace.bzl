"""Loads the FXdiv library, used by XNNPACK & pthreadpool."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "FXdiv",
        strip_prefix = "FXdiv-f8c5354679ec2597792bc70a9e06eff50c508b9a",
        sha256 = "7d3215bea832fe77091ec5666200b91156df6724da1e348205078346325fc45e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/FXdiv/archive/f8c5354679ec2597792bc70a9e06eff50c508b9a.tar.gz",
            "https://github.com/Maratyszcza/FXdiv/archive/f8c5354679ec2597792bc70a9e06eff50c508b9a.tar.gz",
        ],
        build_file = "//third_party/FXdiv:BUILD.bazel",
    )
