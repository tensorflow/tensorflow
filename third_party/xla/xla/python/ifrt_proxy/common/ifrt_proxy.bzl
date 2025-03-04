"""Common libraries for IFRT proxy."""

load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def ifrt_proxy_cc_test(
        **kwargs):
    xla_cc_test(
        **kwargs
    )

default_ifrt_proxy_visibility = ["//xla/python/ifrt_proxy:__subpackages__"]

def cc_library(**attrs):
    native.cc_library(**attrs)
