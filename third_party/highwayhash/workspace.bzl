"""loads the highwayhash library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "highwayhash",
        urls = tf_mirror_urls("https://github.com/google/highwayhash/archive/8e7cfe476f67e865b2be62b5a60a75014a631c9a.tar.gz"),
        sha256 = "9c3e0e87d581feeb0c18d814d98f170ff23e62967a2bd6855847f0b2fe598a37",
        strip_prefix = "highwayhash-8e7cfe476f67e865b2be62b5a60a75014a631c9a",
        build_file = "//third_party/highwayhash:highwayhash.BUILD",
    )
