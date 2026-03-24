"""oneAPI Deep Neural Network Library (oneDNN)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "onednn",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        patch_file = ["//third_party/mkl_dnn:setting_init.patch"],
        sha256 = "071f289dc961b43a3b7c8cbe8a305290a7c5d308ec4b2f586397749abdc88296",
        strip_prefix = "oneDNN-3.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "onednn_async",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        patch_file = ["//third_party/mkl_dnn:setting_init.patch"],
        sha256 = "1cfa18fad65b4c3b46ef701a83c64b87411d63e79c8549cdb37f8c1fc10e2398",
        strip_prefix = "oneDNN-dev-v3.7-thunk-preview",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/heads/dev-v3.7-thunk-preview.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        patch_file = [
            "//third_party/mkl_dnn:onednn_acl_lock_fixed_format_matmul.patch",
            "//third_party/mkl_dnn:onednn_acl_threadpool_default_max.patch",
        ],
        sha256 = "5792cbc07764c6e25c459ff68efb5cfcd7f4a0ba66dca6a4a2c681cd7a644596",
        strip_prefix = "oneDNN-3.7",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.zip"),
    )
