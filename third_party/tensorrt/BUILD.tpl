# -*- python -*-

licenses(["notice"])  

exports_files(["LICENSE"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "if_cuda")

config_setting(
    name = "trt_enabled",
    define_values = {
        "using_tensorrt":"true"
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorrt",
    srcs =[%{tensorrt_lib}],
    hdrs = ["include/NvInfer.h",
            "include/NvUtils.h",
    ],
    copts= cuda_default_copts(),
    deps =["@local_config_cuda//cuda:cuda",
	   "@local_config_cuda//cuda:cudnn",],
    linkstatic = 1,
    #include_prefix="include/",
    includes=["include/"],
    visibility = ["//visibility:public"],	
)

%{tensorrt_genrules}

# filegroup(
#     name = "%{tensorrt_lib}",
#     srcs =  ["%{tensorrt_lib}"],
#     visibility = ["//visibility:public"],
# )
