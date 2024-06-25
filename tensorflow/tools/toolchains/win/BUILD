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
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    remote_execution_properties = """
        properties:{
          name: "container-image"
          value: "docker://gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:1082ef4299a72e44a84388f192ecefc81ec9091c146f507bc36070c089c0edcc"
        }
        properties:{
          name: "OSFamily"
          value: "Windows"
        }
        properties:{
          name: "Pool" value: "default"
        }
        properties:{
          name: "dockerNetwork" value: "off"
        }
        """,
)

# Register the clang-cl platform
platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    remote_execution_properties = """
        properties:{
          name: "container-image"
          value: "docker://gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:1082ef4299a72e44a84388f192ecefc81ec9091c146f507bc36070c089c0edcc"
        }
        properties:{
          name: "OSFamily"
          value: "Windows"
        }
        properties:{
          name: "Pool" value: "default"
        }
        properties:{
          name: "dockerNetwork" value: "off"
        }
        """,
)
