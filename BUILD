exports_files([
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])
# Add a platform target to support clang-cl on Windows
platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)
