"""Common libraries for IFRT proxy."""

load("//xla:xla.bzl", "xla_cc_test")

def ifrt_proxy_cc_test(**kwargs):
    xla_cc_test(**kwargs)

default_ifrt_proxy_visibility = ["//xla/python/ifrt_proxy:__subpackages__"]
