"""Loads OpenCL-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content.
    OPENCL_HEADERS_COMMIT = "0d5f18c6e7196863bc1557a693f1509adfcee056"

    tf_http_archive(
        name = "opencl_headers",
        strip_prefix = "OpenCL-Headers-0d5f18c6e7196863bc1557a693f1509adfcee056",
        sha256 = "03cbc1fd449399be0422cdb021400f63958ef2c5a7c099a0d8f36a705b312f53",
        urls = tf_mirror_urls("https://github.com/KhronosGroup/OpenCL-Headers/archive/{commit}.tar.gz".format(commit = OPENCL_HEADERS_COMMIT)),
        build_file = "//third_party/opencl_headers:opencl_headers.BUILD",
    )
