"""Loads Keras-applications python package."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "keras_applications_archive",
        strip_prefix = "keras-applications-1.0.8",
        sha256 = "7c37f9e9ef93efac9b4956301cb21ce46c474ce9da41fac9a46753bab6823dfc",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/keras-team/keras-applications/archive/1.0.8.tar.gz",
            "https://github.com/keras-team/keras-applications/archive/1.0.8.tar.gz",
        ],
        build_file = "//third_party/keras_applications_archive:BUILD.bazel",
        system_build_file = "//third_party/keras_applications_archive:BUILD.system",
    )
