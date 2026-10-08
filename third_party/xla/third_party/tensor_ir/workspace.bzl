"""Provides the repository macro to import Tensor IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports Tensor IR."""
    TENSOR_IR_COMMIT = "664d385996371db2a8de67b76f82fc89e2470109"
    TENSOR_IR_SHA256 = "846a7bb37ef6a67b0224bd9cb25583ad0d84370a9d2565e3c5a8223c0dcf91f0"

    quilt_http_archive(
        name = "tensor_ir",
        build_file = "//third_party/tensor_ir:tensor_ir.BUILD",
        sha256 = TENSOR_IR_SHA256,
        strip_prefix = "tensor-ir-{}".format(TENSOR_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/tensor-ir/archive/{}.tar.gz".format(TENSOR_IR_COMMIT)),
        series = "//third_party/tensor_ir:patches/series",
    )
