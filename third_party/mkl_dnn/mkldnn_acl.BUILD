exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE": "#undef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
    "#cmakedefine DNNL_WITH_SYCL": "#undef DNNL_WITH_SYCL",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "#undef DNNL_WITH_LEVEL_ZERO",
    "#cmakedefine DNNL_SYCL_CUDA": "#undef DNNL_SYCL_CUDA",
}

template_rule(
    name = "dnnl_config_h",
    src = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = _DNNL_RUNTIME_OMP,
)

template_rule(
    name = "dnnl_version_h",
    src = "include/oneapi/dnnl/dnnl_version.h.in",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "2",
        "@DNNL_VERSION_MINOR@": "3",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "N/A",
    },
)

cc_library(
    name = "mkl_dnn_acl",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/**/*.cpp",
            "src/cpu/*.cpp",
        ],
        exclude = [
            "src/cpu/x64/**",
        ],
    ),
    copts = [
        "-fexceptions",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
    ],
    defines = ["DNNL_AARCH64_USE_ACL=1"],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/aarch64/xbyak_aarch64/src",
        "src/cpu/aarch64/xbyak_aarch64/xbyak_aarch64",
        "src/cpu/gemm",
    ],
    linkopts = ["-lgomp"],
    textual_hdrs = glob(
        [
            "include/**/*",
            "include/*",
            "src/common/*.hpp",
            "src/cpu/**/*.hpp",
            "src/cpu/*.hpp",
            "src/cpu/aarch64/xbyak_aarch64/**/*.h",
        ],
    ) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@compute_library//:arm_compute_graph",
        "@compute_library//:arm_compute_runtime",
    ],
)
