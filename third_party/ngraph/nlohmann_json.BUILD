licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE.MIT"])

cc_library(
    name = "nlohmann_json_lib",
    hdrs = glob([
        "include/nlohmann/**/*.hpp",
    ]),
    copts = [
        "-I external/nlohmann_json_lib",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
