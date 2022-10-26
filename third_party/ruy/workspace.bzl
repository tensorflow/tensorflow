"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "ruy",
        # LINT.IfChange
        sha256 = "a22c42e80c7bb450db8492728e4742ee66f46d5458c45fe67ce2c9b61240630c",
        strip_prefix = "ruy-3286a34cc8de6149ac6844107dfdffac91531e72",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/3286a34cc8de6149ac6844107dfdffac91531e72.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
    )
