"""OneAPI Collective Communication Library (oneCCL)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo_v1():
    tf_http_archive(
        name = "oneccl_v1",
        build_file = "//third_party/oneccl:oneccl_v1.BUILD",
        patch_file = [
            "//third_party/oneccl:ze_loader.patch",
        ],
        sha256 = "016b190557c3a5ee585fe38ce3bf8d6a0c99d7b1a55272083db455b2eff92013",
        strip_prefix = "oneCCL-4ceafd15c03ce46f11eeaf91781a92afebd3cecf",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneCCL/archive/4ceafd15c03ce46f11eeaf91781a92afebd3cecf.tar.gz"),
    )

def repo_v2():
    tf_http_archive(
        name = "oneccl",
        build_file = "//third_party/oneccl:oneccl_v2.BUILD",
        sha256 = "f57ea65a4477003bce367cbed57c06195868d8a3993633a499b38cd0ea165350",
        strip_prefix = "oneCCL-master-v2",
        urls = tf_mirror_urls("https://github.com/uxlfoundation/oneCCL/archive/refs/heads/master-v2.tar.gz"),
    )
