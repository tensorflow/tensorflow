# Description:
#   The Boost library collection (http://www.boost.org)
#
# Most Boost libraries are header-only, in which case you only need to depend
# on :boost. If you need one of the libraries that has a separately-compiled
# implementation, depend on the appropriate libs rule.

# This is only needed for Avro.
package(default_visibility = ["@avro_archive//:__subpackages__"])

licenses(["notice"])  # Boost software license

prefix_dir = "boost_1_61_0"

cc_library(
    name = "boost",
    hdrs = glob([
        prefix_dir + "/boost/**/*.hpp",
        prefix_dir + "/boost/**/*.h",
        prefix_dir + "/boost/**/*.ipp",
    ]),
    includes = [prefix_dir],
)

cc_library(
    name = "filesystem",
    srcs = glob([prefix_dir + "/libs/filesystem/src/*.cpp"]),
    deps = [
        ":boost",
        ":system",
    ],
)

cc_library(
    name = "iostreams",
    srcs = glob([prefix_dir + "/libs/iostreams/src/*.cpp"]),
    deps = [
        ":boost",
        "@bzip2_archive//:bz2lib",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "program_options",
    srcs = glob([prefix_dir + "/libs/program_options/src/*.cpp"]),
    deps = [
        ":boost",
    ],
)

cc_library(
    name = "system",
    srcs = glob([prefix_dir + "/libs/system/src/*.cpp"]),
    deps = [
        ":boost",
    ],
)
