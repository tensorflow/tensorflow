"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "ruy",
        sha256 = "dbfee92fcf9d6a767e9689ca46aa6c1ec4eb8fe69376bacff45dd875226d0ba1",
        strip_prefix = "ruy-38a9266b832767a3f535a74a9e0cf39f7892e594",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/38a9266b832767a3f535a74a9e0cf39f7892e594.zip",
            "https://github.com/google/ruy/archive/38a9266b832767a3f535a74a9e0cf39f7892e594.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
