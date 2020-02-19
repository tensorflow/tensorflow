licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

java_runtime(
    name = "windows_jdk8",
    srcs = [],
    java_home = "C:/openjdk",
)

platform(
    name = "rbe_windows_ltsc2019",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:5e91ddd99345204cd8da2e687d312eb64b3916f257023fd1b651b3dabefd9286",
        "OSFamily": "Windows",
    },
)
