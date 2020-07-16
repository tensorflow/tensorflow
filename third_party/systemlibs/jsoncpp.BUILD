licenses(["unencumbered"])  # Public Domain or MIT

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

HEADERS = [
    "include/json/allocator.h",
    "include/json/assertions.h",
    "include/json/autolink.h",
    "include/json/config.h",
    "include/json/features.h",
    "include/json/forwards.h",
    "include/json/json.h",
    "include/json/reader.h",
    "include/json/value.h",
    "include/json/version.h",
    "include/json/writer.h",
]

genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = """
      for i in $(OUTS); do
        i=$${i##*/}
        ln -sf $(INCLUDEDIR)/jsoncpp/json/$$i $(@D)/include/json/$$i
      done
    """,
)

cc_library(
    name = "jsoncpp",
    hdrs = HEADERS,
    includes = ["."],
    linkopts = ["-ljsoncpp"],
    visibility = ["//visibility:public"],
)
