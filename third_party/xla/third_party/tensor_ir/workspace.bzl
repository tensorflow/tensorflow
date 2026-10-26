"""Provides the repository macro to import Tensor IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports Tensor IR."""
    TENSOR_IR_COMMIT = "0fd8eeda5a9d85219a38c1ea9ec04d66952b85ef"
    TENSOR_IR_SHA256 = "39e1108188c6cd60d8575762c2d4186a56ba1c11269bcacd272437165a315319"

    quilt_http_archive(
        name = "tensor_ir",
        build_file = "//third_party/tensor_ir:tensor_ir.BUILD",
        sha256 = TENSOR_IR_SHA256,
        strip_prefix = "tensor-ir-{}".format(TENSOR_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/tensor-ir/archive/{}.tar.gz".format(TENSOR_IR_COMMIT)),
        series = "//third_party/tensor_ir:patches/series",
    )
