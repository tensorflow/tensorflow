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
          value:"docker://gcr.io/tensorflow-testing/tf-win-rbe@sha256:6bee34693b356baf3b0ba700f4eb27a23d3a04e0c47cbb9de813c61ef30c0b9f"
        }
        properties:{
          name: "OSFamily" value: "Windows"
        }
        """,
)

platform(
    name = "rbe_windows_1803_gpu",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    remote_execution_properties = """
        properties:{
          name:"container-image"
          value:""
        }
        properties:{
          name:"sandbox"
          value:"none"
        }
        properties:{
          name: "OSFamily" value: "Windows"
        }
        """,
)
