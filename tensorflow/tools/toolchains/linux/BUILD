licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

platform(
    name = "remote_linux_aarch64",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "container-image": "docker://us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build-arm64@sha256:3645820ce2d2ae35f48cd274437084ffb158ccaf26fed81241bba6f7d7740e0f",
        "OSFamily": "Linux",
        "Arch": "aarch64",
    },
)

platform(
    name = "linux_armhf",
    constraint_values = [
        "@platforms//cpu:armv7e-mf",
        "@platforms//os:linux",
    ],
)
