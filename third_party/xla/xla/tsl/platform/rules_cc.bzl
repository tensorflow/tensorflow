"""Provides an indirection layer to bazel cc_rules"""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_PLATFORM_RULES_CC_USERS",
)
load(
    "//xla/tsl/platform/default:rules_cc.bzl",
    _cc_binary = "cc_binary",
    _cc_import = "cc_import",
    _cc_library = "cc_library",
    _cc_shared_library = "cc_shared_library",
    _cc_test = "cc_test",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_PLATFORM_RULES_CC_USERS)

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_shared_library = _cc_shared_library
cc_test = _cc_test
