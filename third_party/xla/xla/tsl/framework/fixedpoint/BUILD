# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    features = [
        # Required since headers are not self-contained.
        "-parse_headers",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "fixedpoint",
    hdrs = [
        "FixedPoint.h",
        "MatMatProduct.h",
        "MatMatProductAVX2.h",
        "MatMatProductNEON.h",
        "MatVecProduct.h",
        "PacketMathAVX.h",
        "PacketMathAVX2.h",
        "PacketMathAVX512.h",
        "TypeCastingAVX2.h",
        "TypeCastingAVX512.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/framework:fixedpoint_types",
        "@eigen_archive//:eigen3",
    ],
)

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "FixedPoint.h",
        "MatMatProduct.h",
        "MatMatProductAVX2.h",
        "MatMatProductNEON.h",
        "MatVecProduct.h",
        "PacketMathAVX.h",
        "PacketMathAVX2.h",
        "PacketMathAVX512.h",
        "TypeCastingAVX2.h",
        "TypeCastingAVX512.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow:__subpackages__",
        "//xla/tsl:internal",
    ]),
)
