package(default_visibility = ["//visibility:public"])

prefix_dir = "libpng-1.2.53"

PNG_SOURCES = [
    "png.c",
    "pngerror.c",
    "pngget.c",
    "pngmem.c",
    "pngpread.c",
    "pngread.c",
    "pngrio.c",
    "pngrtran.c",
    "pngrutil.c",
    "pngset.c",
    "pngtrans.c",
    "pngwio.c",
    "pngwrite.c",
    "pngwtran.c",
    "pngwutil.c",
]

genrule(
    name = "configure",
    srcs = glob(
        ["**/*"],
        exclude = [prefix_dir + "/config.h"],
    ),
    outs = [prefix_dir + "/config.h"],
    cmd = "pushd external/png_archive/%s; workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); cp -a * $$workdir; pushd $$workdir; ./configure --enable-shared=no --with-pic=no; popd; popd; cp $$workdir/config.h $(@D); rm -rf $$workdir;" % prefix_dir,
)

cc_library(
    name = "png",
    srcs = [prefix_dir + "/" + source for source in PNG_SOURCES],
    hdrs = glob(["**/*.h"]) + [":configure"],
    includes = [prefix_dir],
    linkopts = ["-lz"],
    visibility = ["//visibility:public"],
)
