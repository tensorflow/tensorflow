# Description:
# Contains script to generate tensorflow/core/util/version_info.cc
# Also contains information about git repository deposited by configure
# in gen/...
package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(
    glob(["gen/*"]) + [
        "gen_git_source.py",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
