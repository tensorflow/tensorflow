licenses(["notice"])  # MIT

JSON_HEADERS = [
    "include/json/assertions.h",
    "include/json/autolink.h",
    "include/json/config.h",
    "include/json/features.h",
    "include/json/forwards.h",
    "include/json/json.h",
    "src/lib_json/json_batchallocator.h",
    "include/json/reader.h",
    "include/json/value.h",
    "include/json/writer.h",
]

JSON_SOURCES = [
    "src/lib_json/json_reader.cpp",
    "src/lib_json/json_value.cpp",
    "src/lib_json/json_writer.cpp",
    "src/lib_json/json_tool.h",
]

INLINE_SOURCES = [
    "src/lib_json/json_valueiterator.inl",
]

cc_library(
    name = "jsoncpp",
    srcs = JSON_SOURCES,
    hdrs = JSON_HEADERS,
    includes = ["include"],
    textual_hdrs = INLINE_SOURCES,
    visibility = ["//visibility:public"],
)
