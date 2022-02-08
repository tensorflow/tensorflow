"""Provides the repository macro to import TensorRT Open Source components."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name = "tensorrt_oss_archive"):
    """Imports TensorRT Open Source Components."""
    TRT_OSS_COMMIT = "42805f078052daad1a98bc5965974fcffaad0960"
    TRT_OSS_SHA256 = "16dd3859664587ed68b456a752298a7de7f9b5ae877818c7dcc054843114b8d2"

    tf_http_archive(
        name = name,
        sha256 = TRT_OSS_SHA256,
        strip_prefix = "TensorRT-{commit}".format(commit = TRT_OSS_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
            "https://github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
        ],
        build_file = "//third_party/tensorrt/plugin:BUILD",
        patch_file = ["//third_party/tensorrt/plugin:tensorrt_oss.patch"],
    )
