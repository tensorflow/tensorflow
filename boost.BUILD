# Description:
#   The Boost library collection (http://www.boost.org)
#
# Most Boost libraries are header-only, in which case you only need to depend
# on :boost. If you need one of the libraries that has a separately-compiled
# implementation, depend on the appropriate libs rule.

# This is only needed for Avro.
package(default_visibility = ["@avro_archive//:__subpackages__"])

licenses(["notice"])  # Boost software license

cc_library(
    name = "boost",
    hdrs = glob([
        "boost/**/*.hpp",
        "boost/**/*.h",
        "boost/**/*.ipp",
    ]),
    includes = ["."],
)

cc_library(
    name = "filesystem",
    srcs = glob(["libs/filesystem/src/*.cpp"]),
    deps = [
        ":boost",
        ":system",
    ],
)

cc_library(
    name = "iostreams",
    srcs = glob(["libs/iostreams/src/*.cpp"]),
    deps = [
        ":boost",
        "@bzip2_archive//:bz2lib",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "program_options",
    srcs = glob(["libs/program_options/src/*.cpp"]),
    deps = [":boost"],
)

cc_library(
    name = "system",
    srcs = glob(["libs/system/src/*.cpp"]),
    deps = [":boost"],
)
