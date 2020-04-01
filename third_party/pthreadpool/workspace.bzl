"""Loads the pthreadpool library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "pthreadpool",
        strip_prefix = "pthreadpool-76042155a8b1e189c8f141429fd72219472c32e1",
        sha256 = "91c7b00c16c60c96f23d1966d524879c0f6044caf4bc5e9fc06518dda643e07e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/pthreadpool/archive/76042155a8b1e189c8f141429fd72219472c32e1.tar.gz",
            "https://github.com/Maratyszcza/pthreadpool/archive/76042155a8b1e189c8f141429fd72219472c32e1.tar.gz",
        ],
        build_file = "//third_party/pthreadpool:BUILD.bazel",
    )
