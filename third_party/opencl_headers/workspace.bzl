"""Loads OpenCL-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "opencl_headers",
        # LINT.IfChange
        strip_prefix = "OpenCL-Headers-1e958b2371f8677215cea877f0abf552efda3723",
        sha256 = "f8102fdb3272f2c2835de62461db96a9f1e91bdb87bfcd22b4b3a1ed0befc961",
        urls = tf_mirror_urls("https://github.com/KhronosGroup/OpenCL-Headers/archive/1e958b2371f8677215cea877f0abf552efda3723.tar.gz"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/opencl_headers.cmake)
        build_file = "//third_party/opencl_headers:opencl_headers.BUILD",
    )
