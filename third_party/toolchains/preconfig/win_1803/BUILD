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
    remote_execution_properties = """
        properties:{
          name:"container-image"
          value:"docker://gcr.io/tensorflow-testing/tf-win-rbe@sha256:f954613b8773930142ac101b6731283bc3a3bc0ef811b7cfe6ae159f412762e7"
        }
        properties:{
          name: "OSFamily" value: "Windows"
        }
        """,
)
