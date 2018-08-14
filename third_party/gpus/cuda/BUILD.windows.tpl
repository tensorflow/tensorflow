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

cc_library(
    name = "cuda_headers",
    hdrs = [
        "cuda/cuda_config.h",
        %{cuda_headers}
    ],
    includes = [
        ".",
        "cuda/include",
        "cuda/include/crt",
    ],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudart_static",
    # /WHOLEARCHIVE:cudart_static.lib will cause a
    # "Internal error during CImplib::EmitThunk" error.
    # Treat this library as interface library to avoid being whole archived when
    # linking a DLL that depends on this.
    # TODO(pcloudy): Remove this rule after b/111278841 is resolved.
    interface_library = "cuda/lib/%{cudart_static_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cuda_driver",
    interface_library = "cuda/lib/%{cuda_driver_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudart",
    interface_library = "cuda/lib/%{cudart_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cublas",
    interface_library = "cuda/lib/%{cublas_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cusolver",
    interface_library = "cuda/lib/%{cusolver_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudnn",
    interface_library = "cuda/lib/%{cudnn_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_header",
    includes = [
        ".",
        "cuda/include",
    ],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cufft",
    interface_library = "cuda/lib/%{cufft_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "curand",
    interface_library = "cuda/lib/%{curand_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    visibility = ["//visibility:public"],
    deps = [
        ":cublas",
        ":cuda_headers",
        ":cudart",
        ":cudnn",
        ":cufft",
        ":curand",
    ],
)

cc_library(
    name = "cupti_headers",
    hdrs = [
        "cuda/cuda_config.h",
        ":cuda-extras",
    ],
    includes = [
        ".",
        "cuda/extras/CUPTI/include/",
    ],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cupti_dsos",
    interface_library = "cuda/lib/%{cupti_lib}",
    system_provided = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libdevice_root",
    data = [":cuda-nvvm"],
    visibility = ["//visibility:public"],
)

%{cuda_include_genrules}
