load("//xla:xla.default.bzl", "xla_cc_test")
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

cc_library(
    name = "cpu_clique_key",
    srcs = ["cpu_clique_key.cc"],
    hdrs = ["cpu_clique_key.h"],
    deps = [
        "//xla/core/collectives:clique_key",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "cpu_clique",
    srcs = ["cpu_clique.cc"],
    hdrs = ["cpu_clique.h"],
    deps = [
        ":cpu_clique_key",
        "//xla/core/collectives:clique",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "cpu_cliques",
    srcs = ["cpu_cliques.cc"],
    hdrs = ["cpu_cliques.h"],
    deps = [
        ":cpu_clique",
        ":cpu_clique_key",
        ":cpu_collectives",
        "//xla:util",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "cpu_cliques_test",
    srcs = ["cpu_cliques_test.cc"],
    deps = [
        ":cpu_clique_key",
        ":cpu_cliques",
        ":in_process_collectives",
        "//xla:util",
        "//xla/core/collectives:rank_id",
        "//xla/service:global_device_id",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cpu_collectives",
    srcs = ["cpu_collectives.cc"],
    hdrs = ["cpu_collectives.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "in_process_collectives",
    srcs = ["in_process_collectives.cc"],
    hdrs = ["in_process_collectives.h"],
    deps = [
        ":cpu_collectives",
        ":in_process_communicator",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:communicator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "in_process_communicator",
    srcs = ["in_process_communicator.cc"],
    hdrs = ["in_process_communicator.h"],
    deps = [
        ":cpu_collectives",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/service:rendezvous",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gloo_kv_store",
    srcs = ["gloo_kv_store.cc"],
    hdrs = ["gloo_kv_store.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = internal_visibility([
        "//xla/pjrt/cpu:legacy_cpu_internal_users",
        "//third_party/py/jax:__subpackages__",
    ]),
    deps = [
        "//xla/pjrt:status_casters",
        "//xla/pjrt/distributed:key_value_store_interface",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@gloo",
    ],
)

cc_library(
    name = "gloo_collectives",
    srcs = ["gloo_collectives.cc"],
    hdrs = ["gloo_collectives.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cpu_collectives",
        ":gloo_communicator",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:communicator",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@gloo",
    ],
)

xla_cc_test(
    name = "gloo_collectives_test",
    srcs = ["gloo_collectives_test.cc"],
    deps = [
        ":cpu_clique_key",
        ":cpu_collectives",
        ":gloo_collectives",
        ":gloo_kv_store",
        "//xla:executable_run_options",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:in_memory_key_value_store",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:collective_ops_utils",
        "//xla/service:global_device_id",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ] + select({
        # Gloo's transport_tcp is not available on MacOS
        "//xla/tsl:macos": [
            "@gloo//:transport_uv",
        ],
        "//conditions:default": [
            "@gloo//:transport_tcp",
        ],
    }),
)

cc_library(
    name = "gloo_communicator",
    srcs = ["gloo_communicator.cc"],
    hdrs = ["gloo_communicator.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cpu_collectives",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@gloo",
    ],
)

cc_library(
    name = "mpi_collectives",
    srcs = ["mpi_collectives.cc"],
    hdrs = ["mpi_collectives.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
        # copybara:uncomment_begin(google-only)
        # "-Ithird_party/openmpi/ompi/include",
        # copybara:uncomment_end
    ],
    features = ["-use_header_modules"],
    deps = [
        ":cpu_collectives",
        ":mpi_communicator",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:communicator",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@mpitrampoline",
    ],
)

cc_library(
    name = "mpi_communicator",
    srcs = ["mpi_communicator.cc"],
    hdrs = ["mpi_communicator.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
        # copybara:uncomment_begin(google-only)
        # "-Ithird_party/openmpi/ompi/include",
        # copybara:uncomment_end
    ],
    features = ["-use_header_modules"],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
        "@mpitrampoline",
    ],
)
