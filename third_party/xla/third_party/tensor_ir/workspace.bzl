"""Provides the repository macro to import Tensor IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports Tensor IR."""
    TENSOR_IR_COMMIT = "05ada823a8495dd042d20691d662bef0ee54dd5d"
    TENSOR_IR_SHA256 = "72d57d8463495f7af7e34296d8b2ce6bb7e9f0ec4d450695ba3f72c30bb9be7f"

    quilt_http_archive(
        name = "tensor_ir",
        build_file = "//third_party/tensor_ir:tensor_ir.BUILD",
        sha256 = TENSOR_IR_SHA256,
        strip_prefix = "tensor-ir-{}".format(TENSOR_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/tensor-ir/archive/{}.tar.gz".format(TENSOR_IR_COMMIT)),
        series = "//third_party/tensor_ir:patches/series",
    )
