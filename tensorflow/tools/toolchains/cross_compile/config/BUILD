package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/ml-devinfra-linux-aarch64-cross-compile@sha256:06040763c500bd2ebaaa4585d4729c88d2c8ccec94baa7fbe9bbe3dc2827d79d",
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
    toolchain = "//tensorflow/tools/toolchains/cross_compile/cc:macos_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
