load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "boost",
        urls = [
            "http://mirror.tensorflow.org/jaist.dl.sourceforge.net/project/boost/boost/1.63.0/boost_1_63_0.tar.gz",
            "https://jaist.dl.sourceforge.net/project/boost/boost/1.63.0/boost_1_63_0.tar.gz",
        ],
        sha256 = "fe34a4e119798e10b8cc9e565b3b0284e9fd3977ec8a1b19586ad1dec397088b",
        strip_prefix = "boost_1_63_0",
        build_file = str(Label("//third_party/boost:boost.BUILD")),
    )
