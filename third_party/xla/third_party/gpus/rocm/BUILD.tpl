# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@config_rocm_hipcc//rocm:build_defs.bzl", "hipcc_config")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_gpu_architectures", "rocm_lib_import", "rocm_version_number")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:private"])

string_flag(
    name = "rocm_path_type",
    build_setting_default = "system",
    values = [
        "hermetic",
        "multiple",
        "system",
        "link_only",
    ],
)

config_setting(
    name = "build_hermetic",
    flag_values = {
        ":rocm_path_type": "hermetic",
    },
)

config_setting(
    name = "multiple_rocm_paths",
    flag_values = {
        ":rocm_path_type": "multiple",
    },
)

config_setting(
    name = "link_only",
    flag_values = {
        ":rocm_path_type": "link_only",
    },
)

config_setting(
    name = "using_hipcc",
    values = {
        "define": "using_rocm_hipcc=true",
    },
)

cc_library(
    name = "config",
    hdrs = [
        "rocm_config/rocm_config.h",
    ],
    include_prefix = "rocm",
    strip_include_prefix = "rocm_config",
)

cc_library(
    name = "config_hermetic",
    hdrs = [
        "rocm_config_hermetic/rocm_config.h",
    ],
    include_prefix = "rocm",
    strip_include_prefix = "rocm_config_hermetic",
)

cc_library(
    name = "rocm_config",
    visibility = ["//visibility:public"],
    deps = select({
        ":build_hermetic": [
            ":config_hermetic",
        ],
        "//conditions:default": [
            "config",
        ],
    }),
)

# This target is required to
# add includes that are used by rocm headers themself
# through the virtual includes
# cleaner solution would be to adjust the xla code
# and remove include prefix that is used to include rocm headers.
cc_library(
    name = "rocm_headers_includes",
    hdrs = glob([
        "%{rocm_root}/include/**",
    ]),
    defines = {"__HIP_DISABLE_CPP_FUNCTIONS__": "1"},
    strip_include_prefix = "%{rocm_root}/include",
    deps = [
        "@xla//third_party/libdrm:drm_headers",
    ],
)

cc_library(
    name = "rocm_headers",
    hdrs = glob([
        "%{rocm_root}/include/**",
        "%{rocm_root}/lib/llvm/lib/**/*.h",
    ]),
    defines = ["MIOPEN_BETA_API=1"],
    include_prefix = "rocm",
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_headers_includes",
    ],
)

cc_library(
    name = "rocm_rpath",
    linkopts = select({
        ":build_hermetic": [
            "-Wl,-rpath,external/%{rocm_repo_name}/rocm/%{rocm_root}/lib",
        ],
        ":link_only": [
        ],
        ":multiple_rocm_paths": [
            "-Wl,-rpath,external/%{rocm_repo_name}/rocm/%{rocm_root}/lib",
            "-Wl,-rpath=%{rocm_lib_paths}",
        ],
        "//conditions:default": [
            "-Wl,-rpath,external/%{rocm_repo_name}/rocm/%{rocm_root}/lib",
            "-Wl,-rpath,/opt/rocm/lib",
        ],
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "hip",
    actual = ":hip_runtime",
    visibility = ["//visibility:public"],
)

rocm_lib_import(
    name = "hip_runtime",
    data = glob(
        [
            "%{rocm_root}/lib/libamdhip64.so*",
            "%{rocm_root}/lib/librocm_kpack.so*",
        ],
    ),
    interface_library = "%{rocm_root}/lib/libamdhip64.so",
    deps = [
        ":amd_comgr_libs",
        ":hiprtc_libs",
        ":hsa_rocr_libs",
        ":rocprofiler_register_libs",
        ":system_libs",
    ],
)

filegroup(
    name = "hsa_rocr_libs_data",
    srcs = glob(["%{rocm_root}/lib/libhsa-runtime64.so*"]),
)

cc_library(
    name = "hsa_rocr_libs",
    data = [":hsa_rocr_libs_data"],
    deps = [
        ":rocprofiler_register_libs",
        ":system_libs",
    ],
)

cc_library(
    name = "hiprtc_libs",
    data = glob(
        [
            "%{rocm_root}/lib/libhiprtc.so*",
            "%{rocm_root}/lib/libhiprtc-builtins.so*",
        ],
    ),
    deps = [
        ":amd_comgr_libs",
        ":hsa_rocr_libs",
    ],
)

cc_library(
    name = "amd_comgr_libs",
    data = glob(
        [
            "%{rocm_root}/lib/libamd_comgr_loader.so*",
            "%{rocm_root}/lib/libamd_comgr.so*",
            "%{rocm_root}/lib/llvm/lib/libLLVM.so*",
            "%{rocm_root}/lib/llvm/lib/libclang-cpp.so*",
        ],
    ),
    deps = [
        ":system_libs",
    ],
)

filegroup(
    name = "rocprofiler_register_libs_data",
    srcs = glob(
        [
            "%{rocm_root}/lib/librocprofiler-register.so*",
        ],
    ),
)

cc_library(
    name = "rocprofiler_register_libs",
    data = [":rocprofiler_register_libs_data"],
)

rocm_lib_import(
    name = "rocblas",
    data = glob([
        "%{rocm_root}/lib/librocblas.so*",
        "%{rocm_root}/lib/rocblas/library/*fallback.dat",
    ]) + glob([
        pattern
        for arch in rocm_gpu_architectures()
        for pattern in [
            "%{rocm_root}/lib/rocblas/library/*" + arch + "*",
            "%{rocm_root}/lib/rocblas/library/" + arch + "/**/*",
            "%{rocm_root}/.kpack/blas_lib_" + arch + ".kpack",
        ]
    ]),
    interface_library = "%{rocm_root}/lib/librocblas.so",
    deps = [
        ":hip_runtime_libs",
        ":hipblaslt_libs",
        ":roctx_libs",
    ],
)

rocm_lib_import(
    name = "hipfft",
    data = glob(["%{rocm_root}/lib/libhipfft.so*"]),
    interface_library = "%{rocm_root}/lib/libhipfft.so",
    deps = [
        ":hip_runtime_libs",
        ":rocfft_libs",
    ],
)

cc_library(
    name = "rocfft_libs",
    data = glob(["%{rocm_root}/lib/librocfft.so*"]) + glob([
        "%{rocm_root}/.kpack/fft_lib_" + arch + ".kpack"
        for arch in rocm_gpu_architectures()
    ]),
    deps = [
        ":hip_runtime_libs",
        ":hiprtc_libs",
    ],
)

rocm_lib_import(
    name = "hiprand",
    data = glob(["%{rocm_root}/lib/libhiprand.so*"]),
    interface_library = "%{rocm_root}/lib/libhiprand.so",
    deps = [
        ":hip_runtime_libs",
        ":rocrand_libs",
    ],
)

cc_library(
    name = "rocrand_libs",
    data = glob(["%{rocm_root}/lib/librocrand.so*"]) + glob([
        "%{rocm_root}/.kpack/rand_lib_" + arch + ".kpack"
        for arch in rocm_gpu_architectures()
    ]),
    deps = [
        ":hip_runtime_libs",
    ],
)

rocm_lib_import(
    name = "miopen",
    data = glob([
        "%{rocm_root}/lib/libMIOpen.so*",
        "%{rocm_root}/share/miopen/**",
        "%{rocm_root}/lib/librocm-core.so*",
    ]),
    interface_library = "%{rocm_root}/lib/libMIOpen.so",
    deps = [
        ":amd_comgr_libs",
        ":hip_runtime_libs",
        ":hipblaslt_libs",
        ":hiprtc_libs",
        ":rocblas_libs",
        ":roctx_libs",
        ":system_libs",
    ],
)

rocm_lib_import(
    name = "rccl",
    data = glob(["%{rocm_root}/lib/librccl.so*"]) + glob([
        "%{rocm_root}/.kpack/rccl_lib_" + arch + ".kpack"
        for arch in rocm_gpu_architectures()
    ]),
    interface_library = "%{rocm_root}/lib/librccl.so",
    deps = [
        ":amdsmi_libs",
        ":hip_runtime_libs",
        ":rocm_smi_libs",
        ":rocprofiler_register_libs",
        ":roctx_libs",
    ],
)

cc_library(
    name = "amdsmi_libs",
    data = glob(["%{rocm_root}/lib/libamd_smi.so*"]),
)

rocm_lib_import(
    name = "rocm_smi",
    data = glob(["%{rocm_root}/lib/librocm_smi64.so*"]),
    interface_library = "%{rocm_root}/lib/librocm_smi64.so",
    deps = [],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocprim",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_headers_includes",
    ],
)

rocm_lib_import(
    name = "hipsparse",
    data = glob(["%{rocm_root}/lib/libhipsparse.so*"]),
    interface_library = "%{rocm_root}/lib/libhipsparse.so",
    deps = [
        ":hip_runtime_libs",
        ":rocsparse_libs",
    ],
)

cc_library(
    name = "rocsparse_libs",
    data = glob(["%{rocm_root}/lib/librocsparse.so*"]) + glob([
        "%{rocm_root}/.kpack/blas_lib_" + arch + ".kpack"
        for arch in rocm_gpu_architectures()
    ]),
    deps = [
        ":hip_runtime_libs",
        ":roctx_libs",
    ],
)

cc_library(
    name = "roctx_libs",
    data = glob([
        "%{rocm_root}/lib/libroctx64.so*",
    ]),
)

rocm_lib_import(
    name = "roctracer",
    data = glob([
        "%{rocm_root}/lib/libroctracer64.so*",
    ]),
    interface_library = "%{rocm_root}/lib/libroctracer64.so",
    deps = [
        ":hsa_rocr_libs",
    ],
)

rocm_lib_import(
    name = "rocprofiler_sdk",
    data = glob(["%{rocm_root}/lib/librocprofiler-sdk*.so*"]),
    interface_library = "%{rocm_root}/lib/librocprofiler-sdk.so",
    deps = [
        ":amd_comgr_libs",
        ":system_libs",
    ],
)

rocm_lib_import(
    name = "rocsolver",
    data = glob([
        "%{rocm_root}/lib/librocsolver.so*",
        "%{rocm_root}/lib/host-math/lib/*.so*",
    ]) + glob([
        "%{rocm_root}/.kpack/blas_lib_" + arch + ".kpack"
        for arch in rocm_gpu_architectures()
    ]),
    interface_library = "%{rocm_root}/lib/librocsolver.so",
    deps = [
        ":hip_runtime_libs",
        ":rocblas_libs",
    ],
)

rocm_lib_import(
    name = "hipsolver",
    data = glob(["%{rocm_root}/lib/libhipsolver.so*"]),
    interface_library = "%{rocm_root}/lib/libhipsolver.so",
    deps = [
        ":hip_runtime_libs",
        ":rocblas_libs",
        ":rocsolver_libs",
        ":rocsparse_libs",
    ],
)

rocm_lib_import(
    name = "hipblas",
    data = glob(["%{rocm_root}/lib/libhipblas.so*"]),
    interface_library = "%{rocm_root}/lib/libhipblas.so",
    deps = [
        ":rocblas_libs",
        ":rocsolver_libs",
    ],
)

rocm_lib_import(
    name = "hipblaslt",
    data = glob(
        [
            "%{rocm_root}/lib/libhipblaslt.so*",
            "%{rocm_root}/lib/librocroller.so*",
        ],
    ) + glob([
        pattern
        for arch in rocm_gpu_architectures()
        for pattern in [
            "%{rocm_root}/lib/hipblaslt/library/*" + arch + "*",
            "%{rocm_root}/lib/hipblaslt/library/" + arch + "/**/*",
        ]
    ]) + glob(
        ["%{rocm_root}/lib/hipblaslt/library/*"],
        exclude = [
            "%{rocm_root}/lib/hipblaslt/library/*gfx*",
        ],
    ),
    interface_library = "%{rocm_root}/lib/libhipblaslt.so",
    deps = [
        ":hip_runtime_libs",
        ":roctx_libs",
    ],
)

filegroup(
    name = "system_libs_data",
    srcs = glob(
        [
            "%{rocm_root}/lib/rocm_sysdeps/lib/*.so*",
            "%{rocm_root}/lib/rocm_sysdeps/share/**",
        ],
        exclude = [
            "%{rocm_root}/lib/rocm_sysdeps/share/terminfo/**",
        ],
    ),
)

cc_library(
    name = "system_libs",
    data = [":system_libs_data"],
)

filegroup(
    name = "toolchain_data",
    srcs = glob(
        include = [
            "%{rocm_root}/bin/hipcc",
            "%{rocm_root}/lib/llvm/bin/*",
            "%{rocm_root}/lib/llvm/lib/clang/*/include/**",
            "%{rocm_root}/lib/llvm/lib/clang/*/lib/**/*.bc",
            "%{rocm_root}/lib/llvm/lib/clang/*/lib/**/*.a",
            "%{rocm_root}/lib/llvm/lib/*.so*",
            "%{rocm_root}/share/hip/version",
            "%{rocm_root}/amdgcn/**",
        ],
        allow_empty = True,
    ) + [":system_libs_data"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rocminfo",
    srcs = [
        "%{rocm_root}/bin/rocminfo",
    ] + [
        ":hsa_rocr_libs_data",
        ":rocprofiler_register_libs_data",
        ":system_libs_data",
    ],
    visibility = ["//visibility:public"],
)

platform(
    name = "linux_x64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://%{rocm_rbe_docker_image}",
        "Pool": "%{rocm_rbe_pool}",
        "OSFamily": "Linux",
    },
)
