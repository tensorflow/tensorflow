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
          value: "docker://gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:188dfe811f7b10a5969a967bff6cf4cb46d600a8c6f596742627a1f9a55842ec"
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
