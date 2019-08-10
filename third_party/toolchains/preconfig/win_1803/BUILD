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
          value:"docker://gcr.io/tensorflow-testing/tf-win-rbe@sha256:e62d7bc6b99d841f47701d2a49c01484699abf7b438a8645c6b3f0d175f0fae2"
        }
        properties:{
          name: "OSFamily" value: "Windows"
        }
        """,
)
