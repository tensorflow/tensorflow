load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@local_xla//third_party/py/rules_pywrap:pywrap.impl.bzl", "python_extension", "pywrap_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_binary.bzl", "py_binary")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-header_modules",
        "-use_header_modules",
    ],
    licenses = ["notice"],
)

py_binary(
    name = "codegen",
    srcs = ["@local_xla//third_party/transformer_engine:codegen.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

genrule(
    name = "make_string_code_utils_cuh",
    srcs = [
        "transformer_engine/common/util/string_header.h.in",
        "transformer_engine/common/utils.cuh",
    ],
    outs = ["string_headers/string_code_utils_cuh.h"],
    cmd = "$(location :codegen) --template_file=$(location transformer_engine/common/util/string_header.h.in) --data_file=$(location transformer_engine/common/utils.cuh) --string_name=string_code_utils_cuh > $@",
    tools = [":codegen"],
)

genrule(
    name = "make_string_code_util_math_h",
    srcs = [
        "transformer_engine/common/util/string_header.h.in",
        "transformer_engine/common/util/math.h",
    ],
    outs = ["string_headers/string_code_util_math_h.h"],
    cmd = "$(location :codegen) --template_file=$(location transformer_engine/common/util/string_header.h.in) --data_file=$(location transformer_engine/common/util/math.h) --string_name=string_code_util_math_h > $@",
    tools = [":codegen"],
)

genrule(
    name = "make_string_code_transpose_rtc_transpose_cu",
    srcs = [
        "transformer_engine/common/util/string_header.h.in",
        "transformer_engine/common/transpose/rtc/transpose.cu",
    ],
    outs = ["string_headers/string_code_transpose_rtc_transpose_cu.h"],
    cmd = "$(location :codegen) --template_file=$(location transformer_engine/common/util/string_header.h.in) --data_file=$(location transformer_engine/common/transpose/rtc/transpose.cu) --string_name=string_code_transpose_rtc_transpose_cu > $@",
    tools = [":codegen"],
)

genrule(
    name = "make_string_code_transpose_rtc_cast_transpose_cu",
    srcs = [
        "transformer_engine/common/util/string_header.h.in",
        "transformer_engine/common/transpose/rtc/cast_transpose.cu",
    ],
    outs = ["string_headers/string_code_transpose_rtc_cast_transpose_cu.h"],
    cmd = "$(location :codegen) --template_file=$(location transformer_engine/common/util/string_header.h.in) --data_file=$(location transformer_engine/common/transpose/rtc/cast_transpose.cu) --string_name=string_code_transpose_rtc_cast_transpose_cu > $@",
    tools = [":codegen"],
)

genrule(
    name = "make_string_code_transpose_rtc_cast_transpose_fusion_cu",
    srcs = [
        "transformer_engine/common/util/string_header.h.in",
        "transformer_engine/common/transpose/rtc/cast_transpose_fusion.cu",
    ],
    outs = ["string_headers/string_code_transpose_rtc_cast_transpose_fusion_cu.h"],
    cmd = "$(location :codegen) --template_file=$(location transformer_engine/common/util/string_header.h.in) --data_file=$(location transformer_engine/common/transpose/rtc/cast_transpose_fusion.cu) --string_name=string_code_transpose_rtc_cast_transpose_fusion_cu > $@",
    tools = [":codegen"],
)

cc_library(
    name = "string_headers",
    hdrs = [
        "string_headers/string_code_transpose_rtc_cast_transpose_cu.h",
        "string_headers/string_code_transpose_rtc_cast_transpose_fusion_cu.h",
        "string_headers/string_code_transpose_rtc_transpose_cu.h",
        "string_headers/string_code_util_math_h.h",
        "string_headers/string_code_utils_cuh.h",
    ],
    includes = ["string_headers"],
)

UNSUPPORTED_ARCHITECTURES_FLAGS = [
    "--no-cuda-gpu-arch=sm_50",
    "--no-cuda-gpu-arch=sm_60",
]

cuda_library(
    name = "nvshmem_api",
    srcs = [
        "transformer_engine/common/nvshmem_api/nvshmem_waitkernel.cu",
        "transformer_engine/common/util/logging.h",
        "transformer_engine/common/util/string.h",
    ],
    hdrs = ["transformer_engine/common/nvshmem_api/nvshmem_waitkernel.h"],
    copts = [
        "-fexceptions",
    ] + UNSUPPORTED_ARCHITECTURES_FLAGS,
    includes = [
        "transformer_engine",
        "transformer_engine/common",
        "transformer_engine/common/include",
        "transformer_engine/common/include/transformer_engine",
    ],
    local_defines = ["NVSHMEM_ENABLE_ALL_DEVICE_INLINING"],
    deps = [
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
        "@local_config_cuda//cuda:nvrtc_headers",
        "@nvshmem//:nvshmem_lib",
    ],
)

cuda_library(
    name = "common_lib",
    srcs = glob(
        [
            "transformer_engine/common/**/*.cc",
            "transformer_engine/common/**/*.cpp",
            "transformer_engine/common/**/*.cu",
        ],
        exclude = [
            "transformer_engine/common/permutation/permutation.cu",
            "transformer_engine/common/transpose/rtc/transpose.cu",
            "transformer_engine/common/transpose/rtc/cast_transpose.cu",
            "transformer_engine/common/transpose/rtc/cast_transpose_fusion.cu",
            "transformer_engine/common/nvshmem_api/nvshmem_waitkernel.cu",
        ],
    ),
    hdrs = glob([
        "transformer_engine/common/**/*.cuh",
        "transformer_engine/common/**/*.h",
    ]),
    copts = [
        "-fexceptions",
        "-Wno-logical-op-parentheses",
        "-Wno-missing-braces",
        "-Wno-pass-failed",
        "-Wno-reorder-ctor",
        "-Wno-unused-variable",
        "-Wno-switch",
        "-Wno-exceptions",
        "-Wno-assume",
        "-Wno-self-assign",
        "-Wno-sometimes-uninitialized",
    ] + UNSUPPORTED_ARCHITECTURES_FLAGS,
    data = ["@local_config_cuda//cuda:cuda_headers"],
    includes = [
        "transformer_engine",
        "transformer_engine/common",
        "transformer_engine/common/include",
        "transformer_engine/common/include/transformer_engine",
        "transformer_engine/common/layer_norm",
    ],
    deps = [
        ":nvshmem_api",
        ":string_headers",
        "@com_google_absl//absl/log",
        "@cuda_nccl//:nccl",
        "@cudnn_frontend_archive//:cudnn_frontend",
        "@cutlass_archive//:cutlass",
        "@local_config_cuda//cuda",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cufft",
        "@local_config_cuda//cuda:cusparse",
        "@local_config_cuda//cuda:nvrtc_headers",
        "@local_tsl//tsl/platform:cuda_root_path",
        "@local_xla//xla/ffi/api:ffi",
        "@pybind11",
    ],
)

cuda_library(
    name = "transformer_engine_jax_utils",
    srcs = ["transformer_engine/jax/csrc/extensions/utils.cpp"],
    hdrs = ["transformer_engine/jax/csrc/extensions/utils.h"],
    copts = [
        "-fexceptions",
    ] + UNSUPPORTED_ARCHITECTURES_FLAGS,
    includes = [
        "transformer_engine",
    ],
    deps = [
        ":common_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)

python_extension(
    name = "transformer_engine_jax_extension",
    srcs = glob(
        [
            "transformer_engine/jax/csrc/*.cpp",
            "transformer_engine/jax/csrc/*.h",
            "transformer_engine/jax/csrc/*/*.h",
            "transformer_engine/jax/csrc/*/*.cpp",
        ],
        exclude = [
            "transformer_engine/jax/csrc/extensions/utils.h",
            "transformer_engine/jax/csrc/extensions/utils.cpp",
        ],
    ),
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
        "-Wno-c++11-narrowing",
    ],
    includes = [
        "transformer_engine",
        "transformer_engine/jax/csrc",
        "transformer_engine/jax/csrc/extensions",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":common_lib",
        ":transformer_engine_jax_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn",
        "@local_xla//xla/ffi/api:c_api",
        "@local_xla//xla/ffi/api:ffi",
        "@pybind11",
    ],
)

pywrap_library(
    name = "transformer_engine_jax",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":transformer_engine_jax_extension",
    ],
)
