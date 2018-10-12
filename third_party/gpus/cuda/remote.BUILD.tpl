# Description:
#   Template for cuda Build file to use a pre-generated config.
licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_nvcc",
    values = {
        "define": "using_cuda_nvcc=true",
    },
)

config_setting(
    name = "using_clang",
    values = {
        "define": "using_cuda_clang=true",
    },
)

# Equivalent to using_clang && -c opt.
config_setting(
    name = "using_clang_opt",
    values = {
        "define": "using_cuda_clang=true",
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

alias(
    name = "cuda_headers",
    actual = "%{remote_cuda_repo}/cuda:cuda_headers",
)

alias(
    name = "cudart_static",
    actual = "%{remote_cuda_repo}/cuda:cudart_static",
)

alias(
    name = "cuda_driver",
    actual = "%{remote_cuda_repo}/cuda:cuda_driver",
)

alias(
    name = "cudart",
    actual = "%{remote_cuda_repo}/cuda:cudart",
)

alias(
    name = "cublas",
    actual = "%{remote_cuda_repo}/cuda:cublas",
)

alias(
    name = "cusolver",
    actual = "%{remote_cuda_repo}/cuda:cusolver",
)

alias(
    name = "cudnn",
    actual = "%{remote_cuda_repo}/cuda:cudnn",
)

alias(
    name = "cudnn_header",
    actual = "%{remote_cuda_repo}/cuda:cudnn_header",
)

alias(
    name = "cufft",
    actual = "%{remote_cuda_repo}/cuda:cufft",
)

alias(
    name = "curand",
    actual = "%{remote_cuda_repo}/cuda:curand",
)

alias(
    name = "cuda",
    actual = "%{remote_cuda_repo}/cuda:cuda",
)

alias(
    name = "cupti_headers",
    actual = "%{remote_cuda_repo}/cuda:cupti_headers",
)

alias(
    name = "cupti_dsos",
    actual = "%{remote_cuda_repo}/cuda:cupti_dsos",
)

alias(
    name = "libdevice_root",
    actual = "%{remote_cuda_repo}/cuda:libdevice_root",
)
