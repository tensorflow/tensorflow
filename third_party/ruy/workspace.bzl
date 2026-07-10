"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "ruy",
        # LINT.IfChange
        sha256 = "64a001a81c05743736ed8695edd521c6093519ef2edb6635ffa142e827cfb86b",
        strip_prefix = "ruy-2af88863614a8298689cc52b1a47b3fcad7be835",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/2af88863614a8298689cc52b1a47b3fcad7be835.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
    )
