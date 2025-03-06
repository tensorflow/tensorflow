licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

java_runtime(
    name = "windows_jdk8",
    srcs = [],
    java_home = "C:/openjdk",
)

# Register a Windows 2022 (Clang) platform.
# Note that while this does support RBE, the current pool size is tiny,
# and this platform is meant to be used as a non-RBE one, for now.
platform(
    name = "windows_ltsc2022_clang",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
    remote_execution_properties = """
        properties:{
          name: "container-image"
	  value: "docker://gcr.io/tensorflow-testing/tf-win2022@sha256:915cb093630432c38b028f56bd31116a5559ebbc688d427b6092d86828ae03bc"
        }
        properties:{
          name: "OSFamily"
          value: "Windows"
        }
        properties:{
          name: "Pool" value: "win2022"
        }
        properties:{
          name: "dockerNetwork" value: "off"
        }
        """,
)
