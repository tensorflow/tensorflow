"""Loads Vulkan-Headers, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "vulkan_headers",
        strip_prefix = "Vulkan-Headers-0e57fc1cfa56a203efe43e4dfb9b3c9e9b105593",
        sha256 = "096c4bff0957e9d6777b47d01c63e99ad9cf9d57e52be688a661b2473f8e52cb",
        urls = [
            "https://mirror.bazel.build/github.com/KhronosGroup/Vulkan-Headers/archive/0e57fc1cfa56a203efe43e4dfb9b3c9e9b105593.tar.gz",
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/0e57fc1cfa56a203efe43e4dfb9b3c9e9b105593.tar.gz",
        ],
        build_file = "//third_party/vulkan_headers:BUILD.bazel",
    )
