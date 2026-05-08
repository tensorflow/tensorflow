# Copyright 2026 The OpenXLA Authors.
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

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")
load("//xla/util:build_defs.bzl", "text_to_binary_proto")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "scratch_space_lookup_table_proto",
    srcs = ["scratch_space_lookup_table.proto"],
    visibility = ["//xla/backends/gpu/libraries/cub:__subpackages__"],
)

cc_library(
    name = "cub_scratch_size_deviceless_lookup",
    srcs = ["cub_scratch_size_deviceless_lookup.cc"],
    hdrs = ["cub_scratch_size_deviceless_lookup.h"],
    visibility = ["//xla:internal"],
    deps = [
        ":cub_sort_utils",
        ":embed_cub_scratch_size_lookup_table",
        ":scratch_space_lookup_table_proto_cc",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/util:file_toc",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/messages:parse_message",
        "@riegeli//riegeli/zstd:zstd_reader",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "cub_sort_utils",
    srcs = ["cub_sort_utils.cc"],
    hdrs = ["cub_sort_utils.h"],
    visibility = ["//xla:internal"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "cub_scratch_size_deviceless_lookup_test",
    srcs = ["cub_scratch_size_deviceless_lookup_test.cc"],
    deps = [
        ":cub_scratch_size_deviceless_lookup",
        ":scratch_space_lookup_table_proto_cc",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/util/proto:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "cub_sort_utils_test",
    srcs = ["cub_sort_utils_test.cc"],
    deps = [
        ":cub_sort_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cuda_library(
    name = "generate_cub_scratch_size_lib",
    testonly = True,
    srcs = ["generate_cub_scratch_size_lib.cu.cc"],
    hdrs = ["generate_cub_scratch_size_lib.cu.h"],
    tags = ["cuda-only"],
    deps = [
        ":scratch_space_lookup_table_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_config_cuda//cuda:cub_headers",  # buildcleaner: keep
        "@tsl//tsl/platform:path",
    ],
)

xla_test(
    name = "generate_cub_scratch_size_test",
    srcs = ["generate_cub_scratch_size_test.cc"],
    backend_tags = {
        "h100": ["full"],
        "b200": ["full"],
        "gb200": ["full"],
    },
    backends = [
        "h100",
        "a100",
        "b200",
        "gb200",
        "v100",
        "p100",
    ],
    tags = [
        "cuda-only",
        "no_oss",  # Stop this test from being run in OSS presubmits
        "notap",
    ],
    deps = [
        ":generate_cub_scratch_size_lib",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

genrule(
    name = "combine_cub_scratch_size_textprotos",
    srcs = glob(["data/*.textproto"]),
    outs = ["cub_scratch_size_lookup_table.textproto"],
    cmd = "cat /dev/null $(SRCS) > $@",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
)

text_to_binary_proto(
    name = "cub_scratch_size_lookup_table_binary",
    src = "cub_scratch_size_lookup_table.textproto",
    out = "cub_scratch_size_lookup_table.binarypb",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    proto_file = ":scratch_space_lookup_table.proto",
    proto_name = "xla.gpu.CubScratchSizeLookupTable",
)

# Copybara can't re-write the path to the OSS version inside the genrule cmd, so we need to alias it.
alias(
    name = "zstd_compressor",
    actual = "//xla/tools:zstd_compressor",
)

# Compress the proto to avoid bloating the binary size.
genrule(
    name = "cub_scratch_size_lookup_table_binary_zstd",
    srcs = [":cub_scratch_size_lookup_table_binary"],
    outs = ["cub_scratch_size_lookup_table.binarypb.zstd"],
    cmd = "$(location :zstd_compressor) --compression_level=22 < $(location :cub_scratch_size_lookup_table_binary) > $@",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tools = [":zstd_compressor"],
)

cc_embed_data(
    name = "embed_cub_scratch_size_lookup_table",
    srcs = [":cub_scratch_size_lookup_table_binary_zstd"],
    outs = [
        "embed_cub_scratch_size_lookup_table.cc",
        "embed_cub_scratch_size_lookup_table.h",
    ],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    embedopts = ["--namespace=xla::gpu"],
)
