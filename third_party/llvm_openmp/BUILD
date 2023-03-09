# Build file for OpenMP library that is part of llvm

load(
    "@org_tensorflow//tensorflow/tsl:tsl.bzl",
    "if_linux_x86_64",
    "if_macos",
    "if_windows",
)
load(
    "@org_tensorflow//third_party/llvm_openmp:cmake_vars.bzl",
    "cmake_var_string",
    "expand_cmake_vars",
)
load(
    "@org_tensorflow//third_party/llvm_openmp:openmp.bzl",
    "dict_add",
    "libiomp5_cc_binary",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

exports_files(["LICENSE.txt"])

py_binary(
    name = "expand_cmake_vars",
    srcs = ["expand_cmake_vars.py"],
    srcs_version = "PY3",
    visibility = [
        "@llvm_openmp//:__subpackages__",
    ],
)

kmp_i18n_os_type = select({
    "@org_tensorflow//tensorflow/tsl:windows": "win",
    "//conditions:default": "lin",
})

genrule(
    name = "kmp_i18n_id",
    srcs = [
        "runtime/tools/message-converter.pl",
        "runtime/tools/lib/tools.pm",
        "runtime/src/i18n/en_US.txt",
    ],
    outs = ["include/kmp_i18n_id.inc"],
    cmd = "perl $(location runtime/tools/message-converter.pl) --os=" + kmp_i18n_os_type + " --prefix=kmp_i18n --enum=$@  $(location runtime/src/i18n/en_US.txt)",
)

genrule(
    name = "kmp_i18n_default",
    srcs = [
        "runtime/tools/message-converter.pl",
        "runtime/tools/lib/tools.pm",
        "runtime/src/i18n/en_US.txt",
    ],
    outs = ["include/kmp_i18n_default.inc"],
    cmd = "perl $(location runtime/tools/message-converter.pl) --os=" + kmp_i18n_os_type + " --prefix=kmp_i18n --default=$@ $(location runtime/src/i18n/en_US.txt)",
)

genrule(
    name = "generate_def",
    srcs = [
        "runtime/tools/generate-def.pl",
    ],
    outs = ["omp_dll.def"],
    cmd = "perl $(location runtime/tools/generate-def.pl) -D arch_32e -D msvc_compat -D norm -o omp_dll.tmp external/llvm_openmp/runtime/src/dllexports; mv omp_dll.tmp $@",
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
    "@org_tensorflow//tensorflow/tsl:windows": cmake_var_string(
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

headers = [
    "runtime/src/kmp_affinity.h",
    "runtime/src/kmp_atomic.h",
    "runtime/src/kmp_debug.h",
    "runtime/src/kmp_dispatch_hier.h",
    "runtime/src/kmp_dispatch.h",
    "runtime/src/kmp_environment.h",
    "runtime/src/kmp_error.h",
    "runtime/src/kmp_ftn_entry.h",
    "runtime/src/kmp_ftn_os.h",
    "runtime/src/kmp_i18n.h",
    "runtime/src/kmp_io.h",
    "runtime/src/kmp_itt.h",
    "runtime/src/kmp_itt.inl",
    "runtime/src/kmp_lock.h",
    "runtime/src/kmp_os.h",
    "runtime/src/kmp_platform.h",
    "runtime/src/kmp_safe_c_api.h",
    "runtime/src/kmp_settings.h",
    "runtime/src/kmp_stats.h",
    "runtime/src/kmp_str.h",
    "runtime/src/kmp_taskdeps.h",
    "runtime/src/kmp_version.h",
    "runtime/src/kmp_wait_release.h",
    "runtime/src/kmp_wrapper_getpid.h",
    "runtime/src/kmp_wrapper_malloc.h",
    "runtime/src/kmp.h",
    "runtime/src/ompt-specific.h",
    "runtime/src/tsan_annotations.h",
]

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
]

common_includes = [
    "runtime/src/",
    "include/",
]

# Expand libiomp build rule based on platform.
if_linux_x86_64(a = libiomp5_cc_binary(
    "libiomp5.so",
    cppsources,
    srcdeps + headers,
    common_includes,
))

if_macos(a = libiomp5_cc_binary(
    "libiomp5.dylib",
    cppsources,
    srcdeps + headers,
    common_includes,
))

if_windows(a = libiomp5_cc_binary(
    "libiomp5md.dll",
    cppsources,
    srcdeps + headers,
    common_includes,
))

bzl_library(
    name = "openmp_bzl",
    srcs = ["openmp.bzl"],
)
