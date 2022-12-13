package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

licenses(["notice"])

cc_library(
    name = "zstd",
    srcs = glob([
        "lib/decompress/zstd*.c",
        "lib/decompress/*_impl.h",
        "lib/compress/zstd*.c",
        "lib/compress/zstd*.h",
    ]) + [
        "lib/decompress/zstd_decompress_block.h",
        "lib/decompress/zstd_decompress_internal.h",
        "lib/decompress/zstd_ddict.h",
        "lib/compress/hist.c",
        "lib/compress/hist.h",
        "lib/common/compiler.h",
        "lib/common/cpu.h",
        "lib/common/bitstream.h",
        "lib/common/entropy_common.c",
        "lib/common/fse_decompress.c",
        "lib/compress/fse_compress.c",
        "lib/compress/huf_compress.c",
        "lib/decompress/huf_decompress.c",
        "lib/common/fse.h",
        "lib/common/huf.h",
        "lib/common/error_private.c",
        "lib/common/zstd_deps.h",
        "lib/common/error_private.h",
        "lib/zstd_errors.h",
        "lib/common/mem.h",
        "lib/common/pool.c",
        "lib/common/pool.h",
        "lib/common/debug.h",
        "lib/common/threading.c",
        "lib/common/threading.h",
        "lib/common/xxhash.c",
        "lib/common/xxhash.h",
        "lib/common/zstd_common.c",
        "lib/common/zstd_internal.h",
        "lib/common/zstd_trace.h",
        "lib/common/debug.c",
    ],
    hdrs = ["lib/zstd.h"],
    copts = [
        "-DZSTD_MULTITHREAD",
        "-DZSTD_DISABLE_ASM",
        "-DXXH_NAMESPACE=ZSTD_",
    ],
    includes = ["lib/"],
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-lpthread",
        ],
    }),
)
