"""Loads Keras-applications python package."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "keras_applications_archive",
        strip_prefix = "keras-applications-1.0.6",
        sha256 = "2cb412c97153160ec267b238e958d281ac3532b139cab42045c2d7086a157c21",
        urls = [
            "http://mirror.tensorflow.org/github.com/keras-team/keras-applications/archive/1.0.6.tar.gz",
            "https://github.com/keras-team/keras-applications/archive/1.0.6.tar.gz",
        ],
        build_file = "//third_party/keras_applications_archive:BUILD.bazel",
        system_build_file = "//third_party/keras_applications_archive:BUILD.system",
    )
