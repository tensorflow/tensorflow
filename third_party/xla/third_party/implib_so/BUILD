package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

py_binary(
    name = "get_symbols",
    srcs = ["get_symbols.py"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        "@implib_so//:implib_gen_lib",
    ],
)

py_binary(
    name = "make_stub",
    srcs = ["make_stub.py"],
    deps = [
        "@bazel_tools//tools/python/runfiles",
        "@implib_so//:implib_gen_lib",
    ],
)
