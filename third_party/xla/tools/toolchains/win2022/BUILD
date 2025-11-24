licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

java_runtime(
    name = "windows_jdk8",
    srcs = [],
    java_home = "C:/openjdk",
)

# Register a Windows 2022 (Clang) platform.
platform(
    name = "windows_ltsc2022_clang",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/tf-win2022@sha256:915cb093630432c38b028f56bd31116a5559ebbc688d427b6092d86828ae03bc",
        "OSFamily": "Windows",
        "Pool": "win2022",
        "dockerNetwork": "off",
        "cache-silo-key": "20251105-1762360217",
    },
)
