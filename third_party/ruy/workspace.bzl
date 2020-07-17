"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "e1b38265ab36662c921be260c68dbe28349a539873baabd974a5140ea64f1fe0",
        strip_prefix = "ruy-d492ac890d982d7a153a326922f362b10de8d2ad",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/d492ac890d982d7a153a326922f362b10de8d2ad.zip",
            "https://github.com/google/ruy/archive/d492ac890d982d7a153a326922f362b10de8d2ad.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
