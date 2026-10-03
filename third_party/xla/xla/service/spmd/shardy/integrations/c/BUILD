# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# XLA SDY C APIs.
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

SDY_CAPI_SOURCES = [
    "passes.cc",
]

SDY_CAPI_HEADERS = [
    "passes.h",
]

cc_library(
    name = "xla_sdy_capi",
    srcs = SDY_CAPI_SOURCES,
    hdrs = SDY_CAPI_HEADERS,
    deps = [
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "@llvm-project//mlir:CAPIIR",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "xla_sdy_capi_headers",
    hdrs = SDY_CAPI_HEADERS,
    includes = ["."],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "xla_sdy_capi_objects",
    srcs = SDY_CAPI_SOURCES,
    hdrs = SDY_CAPI_HEADERS,
    deps = [
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "@llvm-project//mlir:CAPIIRObjects",
    ],
    alwayslink = True,
)
