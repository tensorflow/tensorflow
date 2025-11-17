"""slinky is a lightweight runtime for semi-automatical optimization of data flow pipelines for locality."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "slinky",
        sha256 = "1cf72ab8135680bcefc676eef5acfa59dc68bf7fc5a20e174394edfc85074d08",
        strip_prefix = "slinky-395643708e97085cff91ac5f9d7afc5b4a02f2c5",
        urls = tf_mirror_urls("https://github.com/dsharlet/slinky/archive/395643708e97085cff91ac5f9d7afc5b4a02f2c5.zip"),
    )
