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
        ":clique_id",
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
    hdrs = ["collectives.h"],
    deps = [
        ":clique_id",
        ":communicator",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "communicator",
    hdrs = ["communicator.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "clique_id",
    srcs = ["clique_id.cc"],
    hdrs = ["clique_id.h"],
    deps = [
        "//xla:util",
        "@com_google_absl//absl/crc:crc32c",
        "@com_google_absl//absl/status:statusor",
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
