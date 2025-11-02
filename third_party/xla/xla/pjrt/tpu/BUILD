load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

genrule(
    name = "tpu_versions_proto_generator",
    srcs = if_google(
        ["//learning/brain/research/pjrt/tpu:tpu_versions_internal.proto"],
        ["//xla/pjrt/tpu:tpu_versions_external.proto"],
    ),
    outs = ["tpu_versions.proto"],
    cmd = "cp $< $@ && sed -i 's/xla_\\(external\\|internal\\)/xla/g' $@",
    compatible_with = ["//buildenv/target:non_prod"],
)

tf_proto_library(
    name = "tpu_versions_proto",
    srcs = [":tpu_versions.proto"],
    visibility = internal_visibility(["//xla/pjrt:friends"]),
)

tf_proto_library(
    name = "tpu_versions_external_proto",
    srcs = ["tpu_versions_external.proto"],
    visibility = internal_visibility(["//learning/brain/research/pjrt/tpu:__subpackages__"]),
)

tf_proto_library(
    name = "tpu_platform_type_proto",
    srcs = [":tpu_platform_type.proto"],
    visibility = internal_visibility(["//xla/pjrt:friends"]),
)

cc_library(
    name = "tpu_topology_args",
    hdrs = ["tpu_topology_args.h"],
    visibility = internal_visibility([
        "//xla/pjrt:friends",
        "//learning/brain/research/pjrt/tpu:__subpackages__",
    ]),
    deps = [
        ":tpu_platform_type_proto_cc",
        ":tpu_versions_proto_cc",
        "//xla/pjrt:pjrt_device_dimensions",
    ],
)

cc_library(
    name = "tpu_topology_utils_external",
    srcs = ["tpu_topology_utils_external.cc"],
    hdrs = ["tpu_topology_utils_external.h"],
    deps = [
        ":tpu_topology_args",
        "//xla/pjrt:pjrt_compiler",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "tpu_topology_utils",
    hdrs = ["tpu_topology_utils.h"],
    compatible_with = ["//buildenv/target:libtpu"],
    defines = if_google(["GOOGLE_INTERNAL"]),
    visibility = internal_visibility(["//xla/pjrt:friends"]),
    deps = if_google(
        ["//learning/brain/research/pjrt/tpu:tpu_topology_utils_internal"],
        [":tpu_topology_utils_external"],
    ),
)
