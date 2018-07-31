licenses(["notice"])  # Apache v2

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc",
    linkopts = ["-lgrpc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc++",
    linkopts = ["-lgrpc++"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc_unsecure",
    linkopts = ["-lgrpc_unsecure"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc++_unsecure",
    linkopts = ["-lgrpc++_unsecure"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "ln_grpc_cpp_plugin",
    outs = ["grpc_cpp_plugin.bin"],
    cmd = "ln -s $$(which grpc_cpp_plugin) $@",
)

sh_binary(
    name = "grpc_cpp_plugin",
    srcs = ["grpc_cpp_plugin.bin"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "ln_grpc_python_plugin",
    outs = ["grpc_python_plugin.bin"],
    cmd = "ln -s $$(which grpc_python_plugin) $@",
)

sh_binary(
    name = "grpc_python_plugin",
    srcs = ["grpc_python_plugin.bin"],
    visibility = ["//visibility:public"],
)
