package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/ml-devinfra-linux-aarch64-cross-compile@sha256:11c5ac3b9b4e01cfa82b39b90826a9bfc5b806ccc92cd3d272e6bf861de43be1",
        "OSFamily": "Linux",
    },
)

platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

platform(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

toolchain(
    name = "macos-x86-cross-compile-cc-toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    toolchain = "//tools/toolchains/cross_compile/cc:macos_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
