exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    "#cmakedefine DNNL_WITH_SYCL": "/* #undef DNNL_WITH_SYCL */",
    "#cmakedefine DNNL_WITH_LEVEL_ZERO": "/* #undef DNNL_WITH_LEVEL_ZERO */",
    "#cmakedefine DNNL_SYCL_CUDA": "/* #undef DNNL_SYCL_CUDA */",
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
        "@DNNL_VERSION_MINOR@": "1",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "fbdfeea2642fec05387ed37d565cf904042f507e",
    },
)

cc_library(
    name = "mkl_dnn_acl",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/common/*.hpp",
            "src/cpu/**/*.cpp",
            "src/cpu/**/*.hpp",
        ],
        exclude = ["src/cpu/x64/**/*"],
    ) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    hdrs = glob(["include/*"]),
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
        "src/cpu/gemm",
    ],
    linkopts = ["-lgomp"],
    visibility = ["//visibility:public"],
    deps = [
        "@compute_library//:arm_compute_graph",
        "@compute_library//:arm_compute_runtime",
    ],
)
