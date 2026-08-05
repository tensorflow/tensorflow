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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "random",
    deps = [
        ":distributions",
        ":seed_sequences",
        "//absl/base:endian",
    ],
)

cc_library(
    name = "distributions",
    linkopts = ["-labsl_random_distributions"],
    deps = [
        "//absl/numeric:bits",
        "//absl/numeric:int128",
        "//absl/strings",
    ],
)

cc_library(
    name = "seed_gen_exception",
    linkopts = ["-labsl_random_seed_gen_exception"],
)

cc_library(
    name = "seed_sequences",
    linkopts = [
        "-labsl_random_internal_platform",
        "-labsl_random_internal_pool_urbg",
        "-labsl_random_internal_randen",
        "-labsl_random_internal_randen_hwaes",
        "-labsl_random_internal_randen_hwaes_impl",
        "-labsl_random_internal_randen_slow",
        "-labsl_random_internal_seed_material",
        "-labsl_random_seed_sequences",
        "-pthread",
    ],
    deps = [
        ":seed_gen_exception",
        "//absl/base",
        "//absl/base:endian",
        "//absl/base:raw_logging_internal",
        "//absl/container:inlined_vector",
        "//absl/numeric:int128",
        "//absl/strings",
        "//absl/types:optional",
        "//absl/types:span",
    ],
)

cc_library(
    name = "bit_gen_ref",
)
