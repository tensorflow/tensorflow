# =============================================================================
# LICENSE STATUS: UNRESOLVED — DO NOT MERGE
# =============================================================================
#
# The KDNN library is distributed by Huawei as part of openEuler's KAIL
# BoostKit. The exact license terms have not been verified for inclusion
# in TensorFlow's `third_party/` tree. Until that review is complete,
# this directory ships with `licenses(["restricted"])` so that:
#
#   * Google's internal build tooling refuses to import this directory
#     (it treats `restricted` as a red flag),
#   * external contributors see a clear signal in code review that the
#     license question is open, and
#   * a reviewer cannot accidentally ship this code without addressing
#     the license.
#
# Once the license has been confirmed (expected to be Apache 2.0, but
# not yet verified), this block should be replaced with:
#
#     licenses(["notice"])  # Apache 2.0
#
# and the comment block above removed. See third_party/KDNN/README.md
# for the full discussion.
licenses(["restricted"])  # UNRESOLVED — see block above.

package(default_visibility = ["//visibility:public"])

# =============================================================================
# Configurations
# =============================================================================

# KDNN is enabled only when:
#   1. --define=enable_kdnn=true is passed, AND
#   2. the target platform is Linux aarch64 (Kunpeng).
config_setting(
    name = "enable_kdnn",
    values = {
        "define": "enable_kdnn=true",
    },
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

# =============================================================================
# Public header-only library
# =============================================================================

cc_library(
    name = "kdnn_header",
    hdrs = ["kdnn.h"],
    include_prefix = "third_party/KDNN",
    # No implementation; the actual KDNN library is loaded at runtime
    # via dlopen() — see tensorflow/core/platform/cpu_info.cc.
)

# =============================================================================
# Link-time stub
# =============================================================================
# When kdnn_repository is wired in (via third_party/KDNN/build_defs.bzl),
# the `@kdnn//:kdnn` target provides the actual libkdnn.so. To keep the
# skeleton buildable without a configured KDNN install, we expose a
# BUILD-visible "kdnn" target that resolves to the header-only library.
# A real deployment will substitute @kdnn//:kdnn in kdnn_deps().
cc_library(
    name = "kdnn",
    deps = [":kdnn_header"],
)
