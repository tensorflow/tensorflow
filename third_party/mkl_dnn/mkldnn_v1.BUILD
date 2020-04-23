exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party/mkl_dnn:build_defs.bzl",
    "if_mkl_open_source_only",
    "if_mkl_v1_open_source_only",
)
load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

template_rule(
    name = "dnnl_config_h",
    src = "include/dnnl_config.h.in",
    out = "include/dnnl_config.h",
    substitutions = {
        "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
        "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
        "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
    },
)

# Create the file mkldnn_version.h with MKL-DNN version numbers.
# Currently, the version numbers are hard coded here. If MKL-DNN is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1) Automatically get the version numbers from CMakeLists.txt.

template_rule(
    name = "dnnl_version_h",
    src = "include/dnnl_version.h.in",
    out = "include/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "1",
        "@DNNL_VERSION_MINOR@": "4",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "N/A",
    },
)

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/**/*.cpp",
        "src/cpu/**/*.hpp",
        "src/cpu/xbyak/*.h",
    ]) + if_mkl_v1_open_source_only([
        ":dnnl_config_h",
    ]) + [":dnnl_version_h"],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-DUSE_MKL",
        "-DUSE_CBLAS",
    ] + if_mkl_open_source_only([
        "-UUSE_MKL",
        "-UUSE_CBLAS",
    ]) + if_mkl_v1_open_source_only([
        "-UUSE_MKL",
        "-UUSE_CBLAS",
    ]) + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",  # only works with gcc
        ],
        # TODO(ibiryukov): enable openmp with clang by including libomp as a
        # dependency.
        ":clang_linux_x86_64": [],
        "//conditions:default": [],
    }),
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "@mkl_linux//:mkl_headers",
            "@mkl_linux//:mkl_libs_linux",
        ],
        "@org_tensorflow//tensorflow:macos": [
            "@mkl_darwin//:mkl_headers",
            "@mkl_darwin//:mkl_libs_darwin",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "@mkl_windows//:mkl_headers",
            "@mkl_windows//:mkl_libs_windows",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "mkldnn_single_threaded",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/**/*.cpp",
        "src/cpu/**/*.hpp",
        "src/cpu/xbyak/*.h",
    ]) + [":dnnl_config_h"],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-DMKLDNN_THR=MKLDNN_THR_SEQ",  # Disables threading.
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    visibility = ["//visibility:public"],
)
