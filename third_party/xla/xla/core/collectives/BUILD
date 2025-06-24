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
    name = "clique",
    srcs = ["clique.cc"],
    hdrs = ["clique.h"],
    deps = [
        ":communicator",
        ":rank_id",
        "//xla:util",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "collectives",
    srcs = ["collectives.cc"],
    hdrs = ["collectives.h"],
    deps = [
        ":clique_id",
        ":clique_key",
        ":communicator",
        ":rank_id",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "collectives_registry",
    srcs = ["collectives_registry.cc"],
    hdrs = ["collectives_registry.h"],
    deps = [
        ":collectives",
        "//xla:util",
        "//xla/service:platform_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "communicator",
    hdrs = ["communicator.h"],
    deps = [
        ":rank_id",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service:collective_ops_utils",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "clique_id",
    srcs = ["clique_id.cc"],
    hdrs = ["clique_id.h"],
    deps = [
        "@com_google_absl//absl/crc:crc32c",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "clique_key",
    srcs = ["clique_key.cc"],
    hdrs = ["clique_key.h"],
    deps = [
        ":rank_id",
        "//xla/service:global_device_id",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "rank_id",
    hdrs = ["rank_id.h"],
    deps = [
        "//xla/tsl/lib/gtl:int_type",
    ],
)
