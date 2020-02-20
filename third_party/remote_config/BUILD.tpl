platform(
    name = "platform",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    exec_properties = {
        "container-image": "%{container_image}",
        "Pool": "default",
    },
)
