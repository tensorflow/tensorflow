licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

java_runtime(
    name = "windows_jdk8",
    srcs = [],
    java_home = "C:/openjdk",
)

platform(
    name = "rbe_windows_1803",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/tf-win-rbe@sha256:484be79f239e9d2eac7c388d7717b8f6f57affaf76d83609a2c70821b34ceefb",
        "OSFamily": "Windows",
    },
)

platform(
    name = "rbe_windows_1803_gpu",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    exec_properties = {
        "container-image": "",
        "sandbox": "none",
        "OSFamily": "Windows",
    },
)
