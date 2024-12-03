load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "collectives",
    hdrs = ["collectives.h"],
    deps = [
        ":communicator",
    ],
)

cc_library(
    name = "communicator",
    hdrs = ["communicator.h"],
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
    name = "rank_id",
    hdrs = ["rank_id.h"],
    deps = [
        "//xla/tsl/lib/gtl:int_type",
    ],
)
