package(default_visibility = ["//visibility:private"])

# Create the LICENSE file for libraries that are used by TensorFlow Lite
# C library.
genrule(
    name = "clicenses_generate",
    srcs = [
        "//third_party/eigen3:LICENSE",
        "@arm_neon_2_x86_sse//:LICENSE",
        "@farmhash_archive//:COPYING",
        "@gemmlowp//:LICENSE",
    ],
    outs = ["LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)
