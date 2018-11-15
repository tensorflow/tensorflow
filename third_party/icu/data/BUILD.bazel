package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# Data for core MIME/Unix/Windows encodings:
# ISO 8859-2..9, 15; Windows-125x; EUC-CN; GBK (Windows cp936); GB 18030;
# Big5 (Windows cp950); SJIS (Windows cp932); EUC-JP; EUC-KR, KS C 5601;
# Windows cp949. Data is pre-processed for little-endian platforms. To replicate
# this pre-processing (if you want additional encodings, for example), do the
# following:
#
# First, download, build, and install ICU. This installs tools such as makeconv.
# Then, run the following from your icu4c/source directory:
#   $ cd data/mappings
#   $ rm *.cnv  # there shouldn't be any .cnv files here to begin with
#   $ grep \.ucm ucmcore.mk | \
#     sed 's/\(UCM_SOURCE_CORE=\)\?\([^ ]\+\.ucm\)\\\?/\2/g' | \
#     tr '\n' ' ' | xargs makeconv
#   $ ls *.cnv > filelist.lst
#   $ pkgdata -m common -p ucmcore filelist.lst
#   $ genccode -f custom_conversion_data ucmcore.dat
# This creates custom_conversion_data.c. You will need to change the target
# :conversion_data to depend on your custom source instead of :conversion_data.c
filegroup(
    name = "conversion_files",
    srcs = glob(["icu_conversion_data.c.gz.*"]),
)

# Data files are compressed and split to work around git performance degradation
# around large files.
genrule(
    name = "merge_conversion_data",
    srcs = [":conversion_files"],
    outs = ["conversion_data.c"],
    cmd = "cat $(locations :conversion_files) | gunzip > $@",
)

cc_library(
    name = "conversion_data",
    srcs = [":conversion_data.c"],
    deps = ["@icu//:headers"],
)
