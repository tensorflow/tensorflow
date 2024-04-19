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
          value: "docker://gcr.io/tensorflow-testing/tf-win2019-rbe@sha256:775f2889a35e7cc87f8e1dc83f3195acb4e709cebe3572cf3ae591ccef27e7e8"
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

#Register platform to support clang-cl on Windows
platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)
