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
          value:"docker://gcr.io/tensorflow-testing/tf-rbe-win@sha256:fbc5713566011cc27fc3651183a6e7c2fd56fc6f006618c53f8fc71e742feebd"
        }
        properties:{
          name: "OSFamily" value: "Windows"
        }
        """,
)
