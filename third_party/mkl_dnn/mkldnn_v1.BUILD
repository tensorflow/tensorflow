exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party/mkl_dnn:build_defs.bzl",
    "if_mkl_open_source_only",
    "if_mkldnn_threadpool",
)
load(
    "@org_tensorflow//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
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

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

_DNNL_RUNTIME_THREADPOOL = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

template_rule(
    name = "dnnl_config_h",
    src = "include/dnnl_config.h.in",
    out = "include/dnnl_config.h",
    substitutions = if_mkldnn_threadpool(
        _DNNL_RUNTIME_THREADPOOL,
        if_false = _DNNL_RUNTIME_OMP,
    ),
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
        "@DNNL_VERSION_MINOR@": "5",
        "@DNNL_VERSION_PATCH@": "1",
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
        "src/cpu/x64/jit_utils/jitprofiling/*.c",
        "src/cpu/x64/jit_utils/jitprofiling/*.h",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
    ] + select({
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
    deps = if_mkl_ml(
        ["@org_tensorflow//third_party/mkl:intel_binary_blob"],
        [],
    ),
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
