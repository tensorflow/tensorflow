exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

template_rule(
    name = "mkldnn_config_h",
    src = "include/mkldnn_config.h.in",
    out = "include/mkldnn_config.h",
    substitutions = {
        "#cmakedefine MKLDNN_CPU_BACKEND MKLDNN_BACKEND_${MKLDNN_CPU_BACKEND}": "#define MKLDNN_CPU_BACKEND MKLDNN_BACKEND_NATIVE",
        "#cmakedefine MKLDNN_GPU_BACKEND MKLDNN_BACKEND_${MKLDNN_GPU_BACKEND}": "#define MKLDNN_GPU_BACKEND MKLDNN_BACKEND_NONE",
    },
)

# Create the file mkldnn_version.h with MKL-DNN version numbers.
# Currently, the version numbers are hard coded here. If MKL-DNN is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1) Automatically get the version numbers from CMakeLists.txt.
# TODO(bhavanis): MKL-DNN minor version needs to be updated for MKL-DNN v1.x.
# The current version numbers will work only if MKL-DNN v0.21 is used.

template_rule(
    name = "mkldnn_version_h",
    src = "include/mkldnn_version.h.in",
    out = "include/mkldnn_version.h",
    substitutions = {
        "@MKLDNN_VERSION_MAJOR@": "0",
        "@MKLDNN_VERSION_MINOR@": "21",
        "@MKLDNN_VERSION_PATCH@": "3",
        "@MKLDNN_VERSION_HASH@": "N/A",
    },
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
    ]) + [":mkldnn_version_h"],
    hdrs = glob(["include/*"]),
    copts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": ["-fexceptions"],
    }) + [
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
