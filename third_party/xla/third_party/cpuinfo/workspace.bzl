"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "fe2aa43254838a2eb5658d1742696473a1d834a57f2a0b38d533346bcd212482",
        strip_prefix = "cpuinfo-8ce83db858065145192c97af90cb668ad72a12e9",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/8ce83db858065145192c97af90cb668ad72a12e9.zip"),
    )
