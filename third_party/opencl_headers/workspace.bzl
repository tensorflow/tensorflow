"""Loads OpenCL-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "opencl_headers",
        # LINT.IfChange
        strip_prefix = "OpenCL-Headers-dcd5bede6859d26833cd85f0d6bbcee7382dc9b3",
        sha256 = "ca8090359654e94f2c41e946b7e9d826253d795ae809ce7c83a7d3c859624693",
        urls = tf_mirror_urls("https://github.com/KhronosGroup/OpenCL-Headers/archive/dcd5bede6859d26833cd85f0d6bbcee7382dc9b3.tar.gz"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/opencl_headers.cmake)
        build_file = "//third_party/opencl_headers:opencl_headers.BUILD",
    )
