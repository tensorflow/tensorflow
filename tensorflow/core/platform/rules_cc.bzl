"""Provides an indirection layer to bazel cc_rules"""

load(
    "//tensorflow/core/platform:default/rules_cc.bzl",
    _cc_binary = "cc_binary",
    _cc_import = "cc_import",
    _cc_library = "cc_library",
    _cc_shared_library = "cc_shared_library",
    _cc_test = "cc_test",
)

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_shared_library = _cc_shared_library
cc_test = _cc_test
