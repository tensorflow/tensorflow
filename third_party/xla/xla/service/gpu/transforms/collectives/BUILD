load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "convert_async_collectives_to_sync",
    srcs = ["convert_async_collectives_to_sync.cc"],
    hdrs = ["convert_async_collectives_to_sync.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/collectives:convert_async_collectives_to_sync",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "convert_async_collectives_to_sync_test",
    srcs = ["convert_async_collectives_to_sync_test.cc"],
    deps = [
        ":convert_async_collectives_to_sync",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "collective_backend_assigner",
    srcs = ["collective_backend_assigner.cc"],
    hdrs = ["collective_backend_assigner.h"],
    deps = [
        ":collective_ops_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "collective_backend_assigner_test",
    srcs = ["collective_backend_assigner_test.cc"],
    deps = [
        ":collective_backend_assigner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "collective_ops_utils",
    srcs = ["collective_ops_utils.cc"],
    hdrs = ["collective_ops_utils.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "collective_ops_utils_test",
    srcs = ["collective_ops_utils_test.cc"],
    deps = [
        ":collective_ops_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_collective_combiner_utils",
    srcs = ["gpu_collective_combiner_utils.cc"],
    hdrs = ["gpu_collective_combiner_utils.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:collective_utils",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

xla_cc_test(
    name = "gpu_collective_combiner_utils_test",
    srcs = ["gpu_collective_combiner_utils_test.cc"],
    deps = [
        ":gpu_collective_combiner_utils",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:collective_pipeliner",
        "//xla/service:collective_pipeliner_utils",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_all_gather_combiner",
    srcs = ["all_gather_combiner.cc"],
    hdrs = ["all_gather_combiner.h"],
    deps = [
        ":collective_combiner_annotator",
        ":gpu_collective_combiner_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/collectives:all_gather_combiner",
        "//xla/service:hlo_domain_map",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name =
        "gpu_all_gather_combiner_test",
    srcs = ["all_gather_combiner_test.cc"],
    deps = [
        ":gpu_all_gather_combiner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:collective_utils",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_reduce_scatter_combiner",
    srcs = ["reduce_scatter_combiner.cc"],
    hdrs = ["reduce_scatter_combiner.h"],
    deps = [
        ":collective_combiner_annotator",
        ":gpu_collective_combiner_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_domain_map",
        "//xla/service:reduce_scatter_combiner",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "reduce_scatter_combiner_test",
    srcs = ["reduce_scatter_combiner_test.cc"],
    deps = [
        ":gpu_reduce_scatter_combiner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:collective_utils",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_all_reduce_combiner",
    srcs = ["all_reduce_combiner.cc"],
    hdrs = ["all_reduce_combiner.h"],
    deps = [
        ":collective_combiner_annotator",
        ":gpu_collective_combiner_utils",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/collectives:all_reduce_combiner",
        "//xla/service:hlo_domain_map",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "all_reduce_combiner_test",
    srcs = ["all_reduce_combiner_test.cc"],
    deps = [
        ":gpu_all_reduce_combiner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:collective_utils",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_combiner_annotator",
    srcs = ["collective_combiner_annotator.cc"],
    hdrs = ["collective_combiner_annotator.h"],
    deps = [
        ":collective_ops_utils",
        ":convert_async_collectives_to_sync",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/gpu:gpu_hlo_schedule",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "collective_combiner_annotator_test",
    srcs = ["collective_combiner_annotator_test.cc"],
    deps = [
        ":collective_combiner_annotator",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "all_gather_optimizer",
    srcs = ["all_gather_optimizer.cc"],
    hdrs = ["all_gather_optimizer.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:collective_ops_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "all_gather_optimizer_test",
    srcs = ["all_gather_optimizer_test.cc"],
    deps = [
        ":all_gather_optimizer",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_module_config",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "all_gather_dynamic_slice_simplifier",
    srcs = ["all_gather_dynamic_slice_simplifier.cc"],
    hdrs = ["all_gather_dynamic_slice_simplifier.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/expanders:op_expander_pass",
        "//xla/service:collective_opt_utils",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "all_gather_dynamic_slice_simplifier_test",
    srcs = ["all_gather_dynamic_slice_simplifier_test.cc"],
    deps = [
        ":all_gather_dynamic_slice_simplifier",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "all_reduce_blueconnect",
    srcs = ["all_reduce_blueconnect.cc"],
    hdrs = ["all_reduce_blueconnect.h"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_placer_hdr",
        "//xla/service:global_device_id",
        "//xla/service:hlo_creation_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "all_reduce_blueconnect_test",
    srcs = ["all_reduce_blueconnect_test.cc"],
    deps = [
        ":all_reduce_blueconnect",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/service:computation_placer_hdr",
        "//xla/service:pattern_matcher",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "all_reduce_splitter",
    srcs = ["all_reduce_splitter.cc"],
    hdrs = ["all_reduce_splitter.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:collective_opt_utils",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "all_reduce_splitter_test",
    srcs = ["all_reduce_splitter_test.cc"],
    deps = [
        ":all_reduce_splitter",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu/transforms:reduce_scatter_creator",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "all_reduce_decomposer",
    srcs = ["all_reduce_decomposer.cc"],
    hdrs = ["all_reduce_decomposer.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:collective_ops_utils",
        "//xla/service:shape_inference",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "all_reduce_decomposer_test",
    srcs = ["all_reduce_decomposer_test.cc"],
    deps = [
        ":all_reduce_decomposer",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_runner",
        "//xla/service:platform_util",
        "//xla/tests:hlo_runner_agnostic_test_base",
        "//xla/tests:test_utils",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_select_folder",
    srcs = ["collective_select_folder.cc"],
    hdrs = ["collective_select_folder.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:collective_ops_utils",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "collective_select_folder_test",
    srcs = ["collective_select_folder_test.cc"],
    deps = [
        ":collective_select_folder",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_permute_cycle_decomposer",
    srcs = ["collective_permute_cycle_decomposer.cc"],
    hdrs = ["collective_permute_cycle_decomposer.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:collective_ops_utils",
        "//xla/service:collective_permute_cycle",
        "//xla/service:source_target_pairs",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "collective_permute_cycle_decomposer_test",
    srcs = ["collective_permute_cycle_decomposer_test.cc"],
    deps = [
        ":collective_permute_cycle_decomposer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "collective_send_recv_combiner",
    srcs = ["collective_send_recv_combiner.cc"],
    hdrs = ["collective_send_recv_combiner.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "collective_send_recv_combiner_test",
    srcs = ["collective_send_recv_combiner_test.cc"],
    deps = [
        ":collective_send_recv_combiner",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "async_collective_annotator",
    srcs = ["async_collective_annotator.cc"],
    hdrs = ["async_collective_annotator.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "async_collective_annotator_test",
    srcs = ["async_collective_annotator_test.cc"],
    deps = [
        ":async_collective_annotator",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_query",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)
