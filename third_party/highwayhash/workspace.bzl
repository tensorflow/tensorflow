"""loads the highwayhash library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "highwayhash",
        urls = tf_mirror_urls("https://github.com/google/highwayhash/archive/26731870dbce3e9b0efa97665dd29b52dc0139cb.tar.gz"),
        sha256 = "52d1125c6492d9fa4bfa5115fc46504c397bdedb1896c5c7d7c2dbfd5e6b7f58",
        strip_prefix = "highwayhash-26731870dbce3e9b0efa97665dd29b52dc0139cb",
        build_file = "//third_party/highwayhash:highwayhash.BUILD",
    )
