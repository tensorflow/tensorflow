# Build file for OpenMP library that is part of llvm

load(
    "@org_tensorflow//third_party/llvm:llvm.bzl",
    "cmake_var_string",
    "expand_cmake_vars",
)
load(
    "@org_tensorflow//third_party/llvm_openmp:openmp.bzl",
    "dict_add",
)

exports_files(["LICENSE.txt"])

genrule(
    name = "kmp_i18n_id",
    srcs = [
        "runtime/tools/message-converter.pl",
        "runtime/src/i18n/en_US.txt",
    ],
    outs = ["include/kmp_i18n_id.inc"],
    cmd = "perl $(location runtime/tools/message-converter.pl) --os=lin --prefix=kmp_i18n --enum=$@  $(location runtime/src/i18n/en_US.txt)",
)

genrule(
    name = "kmp_i18n_default",
    srcs = [
        "runtime/tools/message-converter.pl",
        "runtime/src/i18n/en_US.txt",
    ],
    outs = ["include/kmp_i18n_default.inc"],
    cmd = "perl $(location runtime/tools/message-converter.pl) --os=lin --prefix=kmp_i18n --default=$@ $(location runtime/src/i18n/en_US.txt)",
)

# Bazel doesn't accept .txt as an input, rename the ldscript to .inc to workaround.
genrule(
    name = "ldscript",
    srcs = ["runtime/src/exports_so.txt"],
    outs = ["exports_so.inc"],
    cmd = "cp $(location runtime/src/exports_so.txt) $@",
)

genrule(
    name = "openmp_asm",
    srcs = [
        "runtime/src/z_Windows_NT-586_asm.asm",
    ],
    outs = [
        "z_Windows_NT-586_asm.S",
    ],
    cmd = "cp $(location runtime/src/z_Windows_NT-586_asm.asm) $@",
    visibility = ["//visibility:public"],
)

# Common Cmake vars to expand.
omp_vars = {
    "LIBOMP_ENABLE_SHARED": 1,
    "LIBOMP_LEGAL_ARCH": "Intel(R) 64",
    "LIBOMP_LIB_FILE": "libiomp5",
    "LIBOMP_VERSION_MAJOR": 5,
    "LIBOMP_VERSION_MINOR": 0,
}

# Linux Cmake vars to expand.
omp_vars_linux = {
    "LIBOMP_USE_VERSION_SYMBOLS": 1,
    "LIBOMP_HAVE_WEAK_ATTRIBUTE": 1,
    "LIBOMP_USE_ADAPTIVE_LOCKS": 1,
    "LIBOMP_ENABLE_ASSERTIONS": 1,
}

# Windows Cmake vars to expand.
omp_vars_win = {
    "MSVC": 1,
}

omp_all_cmake_vars = select({
    "@org_tensorflow//tensorflow:windows": cmake_var_string(
        dict_add(
            omp_vars,
            omp_vars_win,
        ),
    ),
    "//conditions:default": cmake_var_string(
        dict_add(
            omp_vars,
            omp_vars_linux,
        ),
    ),
})

expand_cmake_vars(
    name = "config_kmp",
    src = "runtime/src/kmp_config.h.cmake",
    cmake_vars = omp_all_cmake_vars,
    dst = "include/kmp_config.h",
)

expand_cmake_vars(
    name = "config_omp",
    src = "runtime/src/include/omp.h.var",
    cmake_vars = omp_all_cmake_vars,
    dst = "include/omp.h",
)

cppsources = [
    "runtime/src/kmp_alloc.cpp",
    "runtime/src/kmp_atomic.cpp",
    "runtime/src/kmp_csupport.cpp",
    "runtime/src/kmp_debug.cpp",
    "runtime/src/kmp_itt.cpp",
    "runtime/src/kmp_environment.cpp",
    "runtime/src/kmp_error.cpp",
    "runtime/src/kmp_global.cpp",
    "runtime/src/kmp_i18n.cpp",
    "runtime/src/kmp_io.cpp",
    "runtime/src/kmp_runtime.cpp",
    "runtime/src/kmp_settings.cpp",
    "runtime/src/kmp_str.cpp",
    "runtime/src/kmp_tasking.cpp",
    "runtime/src/kmp_threadprivate.cpp",
    "runtime/src/kmp_utility.cpp",
    "runtime/src/kmp_barrier.cpp",
    "runtime/src/kmp_wait_release.cpp",
    "runtime/src/kmp_affinity.cpp",
    "runtime/src/kmp_dispatch.cpp",
    "runtime/src/kmp_lock.cpp",
    "runtime/src/kmp_sched.cpp",
    "runtime/src/kmp_taskdeps.cpp",
    "runtime/src/kmp_cancel.cpp",
    "runtime/src/kmp_ftn_cdecl.cpp",
    "runtime/src/kmp_ftn_extra.cpp",
    "runtime/src/kmp_version.cpp",
]

srcdeps = [
    ":config_kmp",
    ":config_omp",
    ":kmp_i18n_id",
    ":kmp_i18n_default",
    ":ldscript",
]

common_includes = [
    "runtime/src/",
    "include/",
]

# TODO(Intel-tf) Replace the following 3 calls to cc_binary with cc_library.
# cc_library should be used for files that are not independently executed. Using
# cc_library results in linking errors. For e.g on Linux, the build fails
# with the following error message.
# ERROR: //tensorflow/BUILD:689:1: Linking of rule '//tensorflow:libtensorflow_framework.so.2.4.0' failed (Exit 1)
# /usr/bin/ld.gold: error: symbol GOMP_parallel_loop_nonmonotonic_guided has undefined version VERSION
# /usr/bin/ld.gold: error: symbol GOMP_parallel_start has undefined version GOMP_1.0
# /usr/bin/ld.gold: error: symbol GOMP_cancellation_point has undefined version GOMP_4.0
# /usr/bin/ld.gold: error: symbol omp_set_num_threads has undefined version OMP_1.0
# ......
# ......

cc_binary(
    name = "libiomp5.so",
    srcs = cppsources + [
        #linux specific files
        "runtime/src/z_Linux_util.cpp",
        "runtime/src/kmp_gsupport.cpp",
        "runtime/src/z_Linux_asm.S",
    ] + srcdeps,
    copts = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
    includes = common_includes,
    linkopts = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
    linkshared = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libiomp5md.dll",
    srcs = cppsources + [
        #window specific files
        "runtime/src/z_Windows_NT_util.cpp",
        "runtime/src/z_Windows_NT-586_util.cpp",
    ] + srcdeps + [":openmp_asm"],
    copts = ["/Domp_EXPORTS /D_M_AMD64 /DOMPT_SUPPORT=0 /D_WINDOWS /D_WINNT /D_USRDLL"],
    includes = common_includes,
    linkopts = ["/MACHINE:X64"],
    linkshared = True,
    visibility = ["//visibility:public"],
)

# MacOS build has not been tested, however since the MacOS build of openmp
# uses the same configuration as Linux, the following should work.
cc_binary(
    name = "libiomp5.dylib",
    srcs = cppsources + [
        #linux/MacOS specific files
        "runtime/src/z_Linux_util.cpp",
        "runtime/src/kmp_gsupport.cpp",
        "runtime/src/z_Linux_asm.S",
    ] + srcdeps,
    copts = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
    includes = common_includes,
    linkopts = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
    linkshared = True,
    visibility = ["//visibility:public"],
)
