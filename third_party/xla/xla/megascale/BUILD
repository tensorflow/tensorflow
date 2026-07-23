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

load("//xla/megascale:package_groups.bzl", "megascale_package_groups")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
        ":__subpackages__",
    ]),
    licenses = ["notice"],
)

megascale_package_groups()

tf_proto_library(
    name = "dcn_topology_proto",
    srcs = ["dcn_topology.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "addresses_proto",
    srcs = ["addresses.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
)

tf_proto_library(
    name = "megascale_runtime_error_overlay_proto",
    srcs = ["megascale_runtime_error_overlay.proto"],
    create_grpc_library = True,
    make_default_target_header_only = True,
)
