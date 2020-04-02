"""Loads the psimd library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "psimd",
        strip_prefix = "psimd-10b4ffc6ea9e2e11668f86969586f88bc82aaefa",
        sha256 = "1fefd66702cb2eb3462b962f33d4fb23d59a55d5889ee6372469d286c4512df4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/Maratyszcza/psimd/archive/10b4ffc6ea9e2e11668f86969586f88bc82aaefa.tar.gz",
            "https://github.com/Maratyszcza/psimd/archive/10b4ffc6ea9e2e11668f86969586f88bc82aaefa.tar.gz",
        ],
        build_file = "//third_party/psimd:BUILD.bazel",
    )
