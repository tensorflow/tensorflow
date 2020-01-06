"""Loads the psimd library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "psimd",
        strip_prefix = "psimd-8fd2884b88848180904a40c452a362d1ee429ad5",
        sha256 = "9d4f05bc5a93a0ab8bcef12027ebe54cfddd0050d4862442449c8de11b4e8c17",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/psimd/archive/8fd2884b88848180904a40c452a362d1ee429ad5.tar.gz",
            "https://github.com/Maratyszcza/psimd/archive/8fd2884b88848180904a40c452a362d1ee429ad5.tar.gz",
        ],
        build_file = "//third_party/psimd:BUILD.bazel",
    )
