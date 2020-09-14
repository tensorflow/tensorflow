licenses(["notice"])  # BSD 2-Clause License

exports_files(["LICENSE"])

cc_library(
    name = "leveldb",
    srcs = [
        "table/block.cc",
        "table/block_builder.cc",
        "table/filter_block.cc",
        "table/format.cc",
        "table/iterator.cc",
        "table/table.cc",
        "table/table_builder.cc",
        "table/two_level_iterator.cc",
        "util/cache.cc",
        "util/coding.cc",
        "util/comparator.cc",
        "util/crc32c.cc",
        "util/env.cc",
        "util/filter_policy.cc",
        "util/hash.cc",
        "util/options.cc",
        "util/status.cc",
    ] + select({
        "//conditions:default": glob([
            "util/env_posix.cc",
        ]),
        "@org_tensorflow//tensorflow:windows": glob([
            "util/env_windows.cc",
        ]),
    }),
    hdrs = [
        "include/leveldb/cache.h",
        "include/leveldb/comparator.h",
        "include/leveldb/env.h",
        "include/leveldb/export.h",
        "include/leveldb/filter_policy.h",
        "include/leveldb/iterator.h",
        "include/leveldb/options.h",
        "include/leveldb/slice.h",
        "include/leveldb/status.h",
        "include/leveldb/table.h",
        "include/leveldb/table_builder.h",
        "port/port.h",
        "port/port_stdcxx.h",
        "port/thread_annotations.h",
        "table/block.h",
        "table/block_builder.h",
        "table/filter_block.h",
        "table/format.h",
        "table/iterator_wrapper.h",
        "table/two_level_iterator.h",
        "util/coding.h",
        "util/crc32c.h",
        "util/env_posix_test_helper.h",
        "util/hash.h",
        "util/logging.h",
        "util/mutexlock.h",
        "util/no_destructor.h",
    ] + select({
        "//conditions:default": glob([
            "util/posix_logger.h",
        ]),
        "@org_tensorflow//tensorflow:windows": glob([
            "util/env_windows_test_helper.h",
            "util/windows_logger.h",
        ]),
    }),
    copts = [],
    defines = [
        "LEVELDB_PLATFORM_POSIX",
        "LEVELDB_IS_BIG_ENDIAN=0",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
