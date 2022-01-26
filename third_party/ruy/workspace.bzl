"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content. 
    RUY_COMMIT = "e6c1b8dc8a8b00ee74e7268aac8b18d7260ab1ce"

    tf_http_archive(
        name = "ruy",
        sha256 = "fa9a0b9041095817bc3533f7b125c3b4044570c0b3ee6c436d2d29dae001c06b",
        strip_prefix = "ruy-e6c1b8dc8a8b00ee74e7268aac8b18d7260ab1ce",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/{commit}.zip".format(commit = RUY_COMMIT)),
        build_file = "//third_party/ruy:BUILD",
    )
