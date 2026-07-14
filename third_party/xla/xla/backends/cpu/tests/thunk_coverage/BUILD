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

filegroup(
    name = "cpu_hlo",
    srcs = ["cpu.hlo"],
)

filegroup(
    name = "host_offload_hlo",
    srcs = ["host_offload.hlo"],
)
