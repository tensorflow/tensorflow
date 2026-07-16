"""Common libraries for IFRT proxy."""

# This file is used in OSS only. It is not transformed by copybara. Therefore all paths in this
# file are OSS paths.

load("@rules_cc//cc:cc_library.bzl", _cc_library = "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

# IMPORTANT: Do not remove this load statement. We rely on that //xla/tsl doesn't exist in g3
# to prevent g3 .bzl files from loading this file.
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def ifrt_proxy_cc_test(
        **kwargs):
    xla_cc_test(
        **kwargs
    )

default_ifrt_proxy_visibility = ["//visibility:public"]

ifrt_proxy_grpc_client_visibility = default_ifrt_proxy_visibility

def cc_library(**attrs):
    _cc_library(**attrs)
