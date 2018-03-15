licenses(["unencumbered"])  # Public Domain or MIT

exports_files(["LICENSE"])

cc_library(
    name = "jsoncpp",
    srcs = [
        "include/json/assertions.h",
        "src/lib_json/json_batchallocator.h",
        "src/lib_json/json_reader.cpp",
        "src/lib_json/json_tool.h",
        "src/lib_json/json_value.cpp",
        "src/lib_json/json_writer.cpp",
    ],
    hdrs = [
        "include/json/autolink.h",
        "include/json/config.h",
        "include/json/features.h",
        "include/json/forwards.h",
        "include/json/json.h",
        "include/json/reader.h",
        "include/json/value.h",
        "include/json/writer.h",
    ],
    copts = ["-DJSON_USE_EXCEPTION=0"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":private"],
)

cc_library(
    name = "private",
    textual_hdrs = ["src/lib_json/json_valueiterator.inl"],
)
