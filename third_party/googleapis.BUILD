# Copyright 2018 Google LLC
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name = "bigtable_protos",
    srcs = [
        "google/api/annotations.proto",
        "google/api/auth.proto",
        "google/api/http.proto",
        "google/bigtable/admin/v2/bigtable_instance_admin.proto",
        "google/bigtable/admin/v2/bigtable_table_admin.proto",
        "google/bigtable/admin/v2/common.proto",
        "google/bigtable/admin/v2/instance.proto",
        "google/bigtable/admin/v2/table.proto",
        "google/bigtable/v2/bigtable.proto",
        "google/bigtable/v2/data.proto",
        "google/iam/v1/iam_policy.proto",
        "google/iam/v1/policy.proto",
        "google/longrunning/operations.proto",
        "google/rpc/error_details.proto",
        "google/rpc/status.proto",
    ],
    include = ".",
    default_runtime = "@protobuf_archive//:protobuf",
    protoc = "@protobuf_archive//:protoc",
    use_grpc_plugin = True,
    deps = ["@protobuf_archive//:cc_wkt_protos"],
)
