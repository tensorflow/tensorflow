# Description:
#   Template for crosstool Build file to use a pre-generated config.
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

alias(
    name = "toolchain",
    actual = "%{remote_cuda_repo}:toolchain",
)
