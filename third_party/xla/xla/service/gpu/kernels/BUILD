# copybara:uncomment_begin(google-only-loads)
# load("//xla/tests:build_defs.bzl", "xla_test")
# load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
# load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")
# copybara:uncomment_end(google-only-loads)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

cc_library(
    name = "custom_fusion",
    srcs = ["custom_fusion.cc"],
    hdrs = ["custom_fusion.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_kernel",
        "//xla:status",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "custom_fusion_pattern",
    srcs = ["custom_fusion_pattern.cc"],
    hdrs = ["custom_fusion_pattern.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "custom_kernel",
    srcs = ["custom_kernel.cc"],
    hdrs = ["custom_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla/stream_executor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# Bundle all custom fusions into a single target, so we can link all fusions and patterns by adding
# a single dependency.
cc_library(
    name = "custom_fusion_library",
    visibility = ["//visibility:public"],
    # copybara:uncomment_begin(google-only)
    # deps = [":cutlass_gemm_fusion"],
    # copybara:uncomment_end(google-only)
)

# copybara:uncomment_begin(google-only)
# # TODO(ezhulenev): We currently do not have a CUTLASS dependency in open source BUILD.
#
# cc_library(
#     name = "cutlass_gemm_fusion",
#     srcs = ["cutlass_gemm_fusion.cc"],
#     hdrs = ["cutlass_gemm_fusion.h"],
#     deps = [
#         ":custom_fusion",
#         ":custom_fusion_pattern",
#         ":custom_kernel",
#         ":cutlass_gemm_kernel",
#         "@com_google_absl//absl/status",
#         "//xla:shape_util",
#         "//xla:status",
#         "//xla:statusor",
#         "//xla:xla_data_proto_cc",
#         "//xla/hlo/ir:hlo",
#         "//xla/service:pattern_matcher",
#         "@local_tsl//tsl/platform:errors",
#         "@local_tsl//tsl/platform:logging",
#         "@local_tsl//tsl/platform:statusor",
#     ],
#     alwayslink = 1,  # static fusion registration
# )
#
# xla_test(
#     name = "cutlass_gemm_fusion_test",
#     srcs = ["cutlass_gemm_fusion_test.cc"],
#     backends = ["gpu"],
#     deps = [
#         ":custom_fusion_pattern",
#         ":cutlass_gemm_fusion",
#         "@com_google_absl//absl/strings",
#         "//xla:debug_options_flags",
#         "//xla:error_spec",
#         "//xla/service/gpu:custom_fusion_rewriter",
#         "//xla/tests:hlo_test_base",
#         "@local_tsl//tsl/platform:test",
#         "@local_tsl//tsl/platform:test_main",
#     ],
# )
#
# cuda_library(
#     name = "cutlass_gemm_kernel",
#     srcs = ["cutlass_gemm_kernel.cu.cc"],
#     hdrs = ["cutlass_gemm_kernel.h"],
#     visibility = ["//visibility:private"],
#     deps = [
#         ":custom_kernel",
#         "//third_party/gpus/cutlass",
#         "//xla:statusor",
#         "//xla:xla_data_proto_cc",
#         "//xla/stream_executor",
#     ],
# )
#
# xla_test(
#     name = "cutlass_gemm_test",
#     srcs = if_cuda_is_configured(["cutlass_gemm_test.cc"]),
#     backends = ["gpu"],
#     deps = [
#         ":cutlass_gemm_kernel",
#         "//xla:types",
#         "//xla:xla_data_proto_cc",
#         "//xla/stream_executor",
#         "//xla/stream_executor:multi_platform_manager",
#         "//xla/stream_executor:platform",
#         "//xla/stream_executor/cuda:cuda_platform",
#         "@local_tsl//tsl/lib/core:status_test_util",
#         "@local_tsl//tsl/platform:status",
#         "@local_tsl//tsl/platform:test",
#         "@local_tsl//tsl/platform:test_benchmark",
#         "@local_tsl//tsl/platform:test_main",
#     ],
# )
#
# copybara:uncomment_end(google-only)
