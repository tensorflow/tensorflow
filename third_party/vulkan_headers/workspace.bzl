"""Loads Vulkan-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "vulkan_headers",
        # LINT.IfChange
        strip_prefix = "Vulkan-Headers-ec2db85225ab410bc6829251bef6c578aaed5868",
        sha256 = "38febe63d53f9c91e90adb1ecd3df0cc0ea834e3a89d96c4fb5961d1cd6dd65e",
        link_files = {
            "//third_party/vulkan_headers:tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc": "tensorflow/vulkan_hpp_dispatch_loader_dynamic.cc",
        },
        urls = tf_mirror_urls("https://github.com/KhronosGroup/Vulkan-Headers/archive/ec2db85225ab410bc6829251bef6c578aaed5868.tar.gz"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/vulkan_headers.cmake)
        build_file = "//third_party/vulkan_headers:vulkan_headers.BUILD",
    )
