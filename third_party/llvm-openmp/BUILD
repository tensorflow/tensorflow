# Build file for OpenMP library that is part of llvm

exports_files(["LICENSE.txt"])

load(
    "@org_tensorflow//third_party/llvm:llvm.bzl",
    "cmake_var_string",
    "dict_add",
    "expand_cmake_vars",
)
load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

genrule(
    name = "il8n_id",
    srcs = [
        "runtime/tools/message-converter.pl",
        "runtime/src/i18n/en_US.txt",
    ],
    outs = ["include/kmp_i18n_id.inc"],
    cmd = "$(location runtime/tools/message-converter.pl) --os=lin --prefix=kmp_i18n --enum=$@  $(location runtime/src/i18n/en_US.txt)",
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

# Cmake vars to replace.
omp_vars = {
    "LIBOMP_USE_VERSION_SYMBOLS": 1,
    "LIBOMP_HAVE_WEAK_ATTRIBUTE": 1,
    "LIBOMP_USE_ADAPTIVE_LOCKS": 1,
    "LIBOMP_ENABLE_ASSERTIONS": 1,
    "LIBOMP_ENABLE_SHARED": 1,
    "LIBOMP_LEGAL_ARCH": "Intel(R) 64",
    "LIBOMP_LIB_FILE": "libiomp5.so",
    "LIBOMP_VERSION_MAJOR": 5,
    "LIBOMP_VERSION_MINOR": 0,
}

omp_all_cmake_vars = cmake_var_string(dict_add(omp_vars))

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

cc_binary(
    name = "libiomp5.so",
    srcs = glob([
        "runtime/src/*.h",
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

        #linux specific files
        "runtime/src/z_Linux_util.cpp",
        "runtime/src/kmp_gsupport.cpp",
        "runtime/src/z_Linux_asm.S",
    ]) + [
        ":config_kmp",
        ":config_omp",
        ":il8n_id",
        ":kmp_i18n_default",
        ":ldscript",
    ],
    copts = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
    includes = [
        "include/",
        "runtime/src/",
    ],
    linkopts = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
    linkshared = True,
    visibility = ["//visibility:public"],
)
