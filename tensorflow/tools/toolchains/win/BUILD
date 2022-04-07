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
    remote_execution_properties = """
        properties:{
          name: "container-image"
          value: "docker://gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:0dcf74c64740f54290138204b7041c027dc2eccf4f2dad3b5320051ba51f03f5"
        }
        properties:{
          name: "OSFamily"
          value: "Windows"
        }
        properties:{
          name: "Pool" value: "default"
        }
        """,
)
