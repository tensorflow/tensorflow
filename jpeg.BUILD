# Description:
#   The Independent JPEG Group's JPEG runtime library.

licenses(["notice"])  # custom notice-style license, see LICENSE

cc_library(
    name = "jpeg",
    srcs = [
        "cderror.h",
        "cdjpeg.h",
        "jaricom.c",
        "jcapimin.c",
        "jcapistd.c",
        "jcarith.c",
        "jccoefct.c",
        "jccolor.c",
        "jcdctmgr.c",
        "jchuff.c",
        "jcinit.c",
        "jcmainct.c",
        "jcmarker.c",
        "jcmaster.c",
        "jcomapi.c",
        "jconfig.h",
        "jcparam.c",
        "jcprepct.c",
        "jcsample.c",
        "jctrans.c",
        "jdapimin.c",
        "jdapistd.c",
        "jdarith.c",
        "jdatadst.c",
        "jdatasrc.c",
        "jdcoefct.c",
        "jdcolor.c",
        "jdct.h",
        "jddctmgr.c",
        "jdhuff.c",
        "jdinput.c",
        "jdmainct.c",
        "jdmarker.c",
        "jdmaster.c",
        "jdmerge.c",
        "jdpostct.c",
        "jdsample.c",
        "jdtrans.c",
        "jerror.c",
        "jfdctflt.c",
        "jfdctfst.c",
        "jfdctint.c",
        "jidctflt.c",
        "jidctfst.c",
        "jidctint.c",
        "jinclude.h",
        "jmemmgr.c",
        "jmemnobs.c",
        "jmemsys.h",
        "jmorecfg.h",
        "jquant1.c",
        "jquant2.c",
        "jutils.c",
        "jversion.h",
        "transupp.h",
    ],
    hdrs = [
        "jerror.h",
        "jpegint.h",
        "jpeglib.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

genrule(
    name = "configure",
    outs = ["jconfig.h"],
    cmd = "cat <<EOF >$@\n" +
          "#define HAVE_PROTOTYPES 1\n" +
          "#define HAVE_UNSIGNED_CHAR 1\n" +
          "#define HAVE_UNSIGNED_SHORT 1\n" +
          "#define HAVE_STDDEF_H 1\n" +
          "#define HAVE_STDLIB_H 1\n" +
          "#ifdef WIN32\n" +
          "#define INLINE __inline\n" +
          "#else\n" +
          "#define INLINE __inline__\n" +
          "#endif\n" +
          "EOF\n",
)
