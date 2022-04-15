"""Provides the repository macro to import TensorRT Open Source components."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name = "tensorrt_oss_archive"):
    """Imports TensorRT Open Source Components."""
    TRT_OSS_COMMIT = "9ec6eb6db39188c9f3d25f49c8ee3a9721636b56"
    TRT_OSS_SHA256 = "4fa2a712a5f2350b81df01d55c1dc17451e09efd4b2a53322b0433721009e1c7"

    tf_http_archive(
        name = name,
        sha256 = TRT_OSS_SHA256,
        strip_prefix = "TensorRT-{commit}".format(commit = TRT_OSS_COMMIT),
        urls = [
            # TODO: Google Mirror "https://storage.googleapis.com/...."
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
            "https://github.com/NVIDIA/TensorRT/archive/{commit}.tar.gz".format(commit = TRT_OSS_COMMIT),
        ],
        build_file = "//third_party/tensorrt/plugin:BUILD",
        patch_file = ["//third_party/tensorrt/plugin:tensorrt_oss.patch"],
    )
