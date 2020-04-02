platform(
    name = "platform",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:%{platform}",
    ],
    exec_properties = %{exec_properties},
)
