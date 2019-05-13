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
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
)

# Provides CUDA headers for '#include "third_party/gpus/cuda/include/cuda.h"'
# All clients including TensorFlow should use these directives.
cc_library(
    name = "cuda_virtual_headers",
    hdrs = [
        "cuda/cuda_config.h",
        ":cuda-include"
    ],
    include_prefix = "third_party/gpus",
    visibility = ["//visibility:private"],
)

# Provides CUDA headers for '#include <cuda.h>'.
# CUDA itself as well as Eigen use these directives.
cc_library(
    name = "cuda_headers",
    textual_hdrs = [
        # TODO(csigg): change references to third_party/gpus/cuda/cuda_config.h
        # (e.g. in the PIP build script) and then remove cuda_config.h.
        "cuda/cuda_config.h",
        ":cuda-include"
    ],
    includes = [
        ".",  # required to include cuda/cuda/cuda_config.h as cuda/config.h
        "cuda/include",
    ],
)
    deps = [":cuda_virtual_headers"],
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
)

cc_import(
    name = "cuda_driver",
    interface_library = "cuda/lib/%{cuda_driver_lib}",
    system_provided = 1,
)

cc_import(
    name = "cudart",
    interface_library = "cuda/lib/%{cudart_lib}",
    system_provided = 1,
)

cc_library(
    name = "cublas_virtual_headers",
    hdrs = [":cublas-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "cublas/include",
    visibility = ["//visibility:private"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cublas_headers",
    textual_hdrs = [":cublas-include"],
    includes = ["cublas/include"],
    deps = [":cublas_virtual_headers"],
)

cc_import(
    name = "cublas",
    interface_library = "cuda/lib/%{cublas_lib}",
    system_provided = 1,
)

cc_import(
    name = "cusolver",
    interface_library = "cuda/lib/%{cusolver_lib}",
    system_provided = 1,
)

cc_import(
    name = "cudnn",
    interface_library = "cuda/lib/%{cudnn_lib}",
    system_provided = 1,
)

cc_library(
    name = "cudnn_header",
    hdrs = [":cudnn-include"],
    include_prefix = "third_party/gpus/cudnn",
    strip_include_prefix = "cudnn/include",
    deps = [":cuda_headers"],
)

cc_import(
    name = "cufft",
    interface_library = "cuda/lib/%{cufft_lib}",
    system_provided = 1,
)

cc_import(
    name = "curand",
    interface_library = "cuda/lib/%{curand_lib}",
    system_provided = 1,
)

cc_library(
    name = "cuda",
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
    name = "cupti_virtual_headers",
    hdrs = [":cuda-extras"],
    include_prefix="third_party/gpus",
    visibility = ["//visibility:private"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cupti_headers",
    textual_hdrs = [":cuda-extras"],
    includes = ["cuda/extras/CUPTI/include/"],
    deps = [":cupti_virtual_headers"],
)

cc_import(
    name = "cupti_dsos",
    interface_library = "cuda/lib/%{cupti_lib}",
    system_provided = 1,
)

cc_import(
    name = "cusparse",
    interface_library = "cuda/lib/%{cusparse_lib}",
    system_provided = 1,
)

cc_library(
    name = "libdevice_root",
    data = [":cuda-nvvm"],
)

%{copy_rules}
