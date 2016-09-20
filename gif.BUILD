SOURCES = [
    "dgif_lib.c",
    "egif_lib.c",
    "gif_font.c",
    "gif_hash.c",
    "gifalloc.c",
    "openbsd-reallocarray.c",
    "gif_err.c",
    "quantize.c",
]

HEADERS = [
    "gif_hash.h",
    "gif_lib.h",
    "gif_lib_private.h",
]

config_setting(
        name = "windows",
        values = {
            "cpu": "x64_windows_msvc",
        },
        visibility = ["//visibility:public"],
)

prefix_dir = "giflib-5.1.4/lib"
prefix_dir_windows = "windows/giflib-5.1.4/lib"

genrule(
  name = "srcs_without_unistd",
  srcs = [prefix_dir + "/" + source for source in SOURCES],
  outs = [prefix_dir_windows + "/" + source for source in SOURCES],
  cmd = "for f in $(SRCS); do " +
        "  sed 's/#include <unistd.h>//g' $$f > $(@D)/%s/$$(basename $$f);" % prefix_dir_windows +
        "done",
)

genrule(
  name = "hdrs_without_unistd",
  srcs = [prefix_dir + "/" + hdrs for hdrs in HEADERS],
  outs = [prefix_dir_windows + "/" + hdrs for hdrs in HEADERS],
  cmd = "for f in $(SRCS); do " +
        "  sed 's/#include <unistd.h>//g' $$f > $(@D)/%s/$$(basename $$f);" % prefix_dir_windows +
        "done",
)

cc_library(
    name = "gif",
    srcs = select({
        "//conditions:default" : [prefix_dir + "/" + source for source in SOURCES],
        ":windows" : [":srcs_without_unistd"],
    }),
    hdrs = select({
        "//conditions:default" : [prefix_dir + "/" + hdrs for hdrs in HEADERS],
        ":windows" : [":hdrs_without_unistd"],
    }),
    includes = select({
        "//conditions:default" : [prefix_dir],
        ":windows" : [prefix_dir_windows],
    }),
    defines = [
        "HAVE_CONFIG_H",
    ],
    visibility = ["//visibility:public"],
)
