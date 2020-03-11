"""Loads OpenCL-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "opencl_headers",
        strip_prefix = "OpenCL-Headers-0d5f18c6e7196863bc1557a693f1509adfcee056",
        sha256 = "03cbc1fd449399be0422cdb021400f63958ef2c5a7c099a0d8f36a705b312f53",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/KhronosGroup/OpenCL-Headers/archive/0d5f18c6e7196863bc1557a693f1509adfcee056.tar.gz",
            "https://github.com/KhronosGroup/OpenCL-Headers/archive/0d5f18c6e7196863bc1557a693f1509adfcee056.tar.gz",
        ],
        build_file = "//third_party/opencl_headers:BUILD.bazel",
    )
