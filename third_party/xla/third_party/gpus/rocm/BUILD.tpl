load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_version_number", "select_threshold")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:private"])

string_flag(
    name = "rocm_path_type",
    build_setting_default = "system",
    values = [
       "hermetic",
       "multiple",
       "system",
    ]
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
    strip_include_prefix = "%{rocm_root}/include",
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
        ":rocm_rpath",
        ":rocm_headers_includes",
    ],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":hip",
        ":hipblas",
        ":hipblaslt",
        ":hiprand",
        ":hipsolver",
        ":hipsparse",
        ":hsa_rocr",
        ":miopen",
        ":rocblas",
        ":rocm_config",
        ":rocprofiler_register",
        ":rocsolver",
        ":rocsparse",
        ":roctracer",
    ] + select_threshold(
        above_or_eq = [":hipfft"],
        below = [":rocfft"],
        threshold = 40100,
        value = rocm_version_number(),
    ),
)

cc_library(
    name = "hsa_rocr",
    srcs = glob(["%{rocm_root}/lib/libhsa-runtime*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hsa/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    deps = [":rocm_config"],
)

cc_library(
    name = "rocm_rpath",
    linkopts = select({
        ":build_hermetic": [
            "-Wl,-rpath,%{rocm_toolkit_path}/lib",
        ],
        ":multiple_rocm_paths": [
            "-Wl,-rpath=%{rocm_lib_paths}",
        ],
        "//conditions:default": [
            "-Wl,-rpath,/opt/rocm/lib",
        ],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hip",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_hip",
        ":rocm_rpath",
    ],
)

cc_library(
    name = "rocm_hip",
    srcs = glob(["%{rocm_root}/lib/libamdhip*.so"]),
    hdrs = glob(["%{rocm_root}/include/hip/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    strip_include_prefix = "%{rocm_root}",
    deps = [
        ":amd_comgr",
        ":hsa_rocr",
        ":rocm_config",
        ":rocm_smi",
        ":rocprofiler_register",
        ":system_libs",
    ],
)

cc_library(
    name = "rocblas",
    hdrs = glob(["%{rocm_root}/include/rocblas/**"]),
    data = glob([
        "%{rocm_root}/lib/librocblas*.so*",
        "%{rocm_root}/lib/rocblas/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    # workaround to  bring tensile files to the same fs layout as expected in the lib
    # rocblas assumes that tensile files are located in ../roblas/libraries directory
    linkopts = ["-Wl,-rpath,local_config_rocm/rocm/rocm_dis/lib"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocfft",
    srcs = glob(["%{rocm_root}/lib/librocfft*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hipfft",
    srcs = glob(["%{rocm_root}/lib/libhipfft*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hiprand",
    srcs = glob(["%{rocm_root}/lib/libhiprand*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hiprand/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
        "%{rocm_root}/include/rocrand",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "miopen",
    hdrs = glob(["%{rocm_root}/include/miopen/**"]),
    data = glob([
        "%{rocm_root}/lib/libMIOpen*.so*",
        "%{rocm_root}/share/miopen/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    # workaround to  bring miopen db files to the same fs layout as expected in the lib
    # rocblas assumes that miopen db files are located in ../share/miopen/db directory
    linkopts = ["-Wl,-rpath,local_config_rocm/rocm/rocm_dis/lib"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rccl",
    srcs = glob(["%{rocm_root}/lib/librccl*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rccl/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkopts = ["-lnuma"],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
      ":rocm_config",
      ":system_libs",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocprim",
    srcs = [
        "%{rocm_root}/include/hipcub/hipcub_version.hpp",
        "%{rocm_root}/include/rocprim/rocprim_version.hpp",
    ],
    hdrs = glob([
        "%{rocm_root}/include/hipcub/**",
        "%{rocm_root}/include/rocprim/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/hipcub",
        "%{rocm_root}/include/rocprim",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
        ":rocm_headers",
    ],
)

cc_library(
    name = "hipsparse",
    srcs = glob(["%{rocm_root}/lib/libhipsparse*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hipsparse/**"]),
    data = glob(["%{rocm_root}/lib/libhipsparse*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "roctracer",
    hdrs = glob(["%{rocm_root}/include/roctracer/**"]),
    data = glob(["%{rocm_root}/lib/libroctracer*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocsolver",
    srcs = glob(["%{rocm_root}/lib/librocsolver*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rocsolver/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocsparse",
    srcs = glob(["%{rocm_root}/lib/librocsparse*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hipsolver",
    srcs = glob(["%{rocm_root}/lib/libhipsolver*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hipsolver/**"]),
    data = glob(["%{rocm_root}/lib/libhipsolver*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hipblas",
    srcs = glob(["%{rocm_root}/lib/libhipblas.so*"]),
    hdrs = glob(["%{rocm_root}/include/hipblas/**"]),
    data = glob(["%{rocm_root}/lib/libhipblas.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hipblaslt",
    hdrs = glob(["%{rocm_root}/include/hipblaslt/**"]),
    data = glob([
        "%{rocm_root}/lib/hipblaslt/**",
        "%{rocm_root}/lib/libhipblaslt.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    # workaround to  bring tensile files to the same fs layout as expected in the lib
    # hibplatslt assumes that tensile files are located in ../hipblaslt/libraries directory
    linkopts = ["-Wl,-rpath,local_config_rocm/rocm/rocm_dis/lib"],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocrand",
    srcs = glob(["%{rocm_root}/lib/librocrand*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rocrand/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocprofiler_register",
    srcs = glob([
        "%{rocm_root}/lib/librocprofiler-register.so*",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    strip_include_prefix = "%{rocm_root}",
    deps = [":rocm_config"],
)

cc_library(
    name = "amd_comgr",
    srcs = glob([
        "%{rocm_root}/lib/libamd_comgr.so*",
    ]),
    hdrs = glob(["%{rocm_root}/include/amd_comgr/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    strip_include_prefix = "%{rocm_root}",
    deps = [":rocm_config"],
)

cc_library(
    name = "rocm_smi",
    srcs = glob([
        "%{rocm_root}/lib/librocm_smi64.so*",
        "%{rocm_root}/lib/libroam.so*",
    ]),
    hdrs = glob([
        "%{rocm_root}/include/oam/**",
        "%{rocm_root}/include/rocm_smi/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    strip_include_prefix = "%{rocm_root}",
    deps = [":rocm_config"],
)

cc_library(
    name = "system_libs",
    srcs = glob([
        "rocm_dist/usr/lib/**/libelf.so*",
        "rocm_dist/usr/lib/**/libdrm.so*",
        "rocm_dist/usr/lib/**/libnuma.so*",
        "rocm_dist/usr/lib/**/libdrm_amdgpu.so*",
    ]),
    data = glob([
        "rocm_dist/usr/**",
    ]),
)

filegroup(
    name = "rocm_root",
    srcs = [
        "%{rocm_root}/bin/clang-offload-bundler",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(["%{rocm_root}/**"]),
    visibility = ["//visibility:public"],
)
