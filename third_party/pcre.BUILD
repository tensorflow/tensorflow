licenses(["notice"])  # BSD

exports_files(["COPYING"])

cc_library(
    name = "pcre",
    srcs = [
        "pcre_byte_order.c",
        "pcre_chartables.c",
        "pcre_compile.c",
        "pcre_config.c",
        "pcre_dfa_exec.c",
        "pcre_exec.c",
        "pcre_fullinfo.c",
        "pcre_get.c",
        "pcre_globals.c",
        "pcre_internal.h",
        "pcre_jit_compile.c",
        "pcre_maketables.c",
        "pcre_newline.c",
        "pcre_ord2utf8.c",
        "pcre_refcount.c",
        "pcre_string_utils.c",
        "pcre_study.c",
        "pcre_tables.c",
        "pcre_ucd.c",
        "pcre_valid_utf8.c",
        "pcre_version.c",
        "pcre_xclass.c",
        "ucp.h",
    ],
    hdrs = [
        "pcre.h",
        "pcreposix.h",
    ],
    copts = [
        "-DHAVE_BCOPY=1",
        "-DHAVE_INTTYPES_H=1",
        "-DHAVE_MEMMOVE=1",
        "-DHAVE_STDINT_H=1",
        "-DHAVE_STRERROR=1",
        "-DHAVE_SYS_STAT_H=1",
        "-DHAVE_SYS_TYPES_H=1",
        "-DHAVE_UNISTD_H=1",
        "-DLINK_SIZE=2",
        "-DMATCH_LIMIT=10000000",
        "-DMATCH_LIMIT_RECURSION=1000",
        "-DMAX_NAME_COUNT=10000",
        "-DMAX_NAME_SIZE=32",
        "-DNEWLINE=10",
        "-DNO_RECURSE",
        "-DPARENS_NEST_LIMIT=50",
        "-DPCRE_STATIC=1",
        "-DPOSIX_MALLOC_THRESHOLD=10",
        "-DSTDC_HEADERS=1",
        "-DSUPPORT_UCP",
        "-DSUPPORT_UTF",
    ],
    includes = ["."],
    visibility = ["@swig//:__pkg__"],  # Please use RE2
    alwayslink = 1,
)

genrule(
    name = "pcre_h",
    srcs = ["pcre.h.in"],
    outs = ["pcre.h"],
    cmd = "sed -e s/@PCRE_MAJOR@/8/" +
          "    -e s/@PCRE_MINOR@/39/" +
          "    -e s/@PCRE_PRERELEASE@//" +
          "    -e s/@PCRE_DATE@/redacted/" +
          "    $< >$@",
)

genrule(
    name = "pcre_chartables_c",
    srcs = ["pcre_chartables.c.dist"],
    outs = ["pcre_chartables.c"],
    cmd = "cp $< $@",
)
