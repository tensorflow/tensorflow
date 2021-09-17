"""This file contains BUILD extensions for building llvm_openmp.
TODO(Intel-tf): Delete this and reuse a similar function in third_party/llvm
after the TF 2.4 branch cut has passed.
"""

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
)

WINDOWS_MSVC_LLVM_OPENMP_LIBPATH = "bazel-out/x64_windows-opt/bin/external/llvm_openmp/libiomp5md.dll.if.lib"
WINDOWS_MSVC_LLVM_OPENMP_LINKOPTS = "/NODEFAULTLIB:libomp /DEFAULTLIB:" + WINDOWS_MSVC_LLVM_OPENMP_LIBPATH

def windows_llvm_openmp_linkopts():
    return WINDOWS_MSVC_LLVM_OPENMP_LINKOPTS

def dict_add(*dictionaries):
    """Returns a new `dict` that has all the entries of the given dictionaries.

    If the same key is present in more than one of the input dictionaries, the
    last of them in the argument list overrides any earlier ones.

    Args:
      *dictionaries: Zero or more dictionaries to be added.

    Returns:
      A new `dict` that has all the entries of the given dictionaries.
    """
    result = {}
    for d in dictionaries:
        result.update(d)
    return result

def select_os_specific(L, M, W):
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": L,
        "@org_tensorflow//tensorflow:macos": M,
        "@org_tensorflow//tensorflow:windows": W,
        "//conditions:default": L,
    })

def select_os_specific_2(LM, W):
    return select_os_specific(L = LM, M = LM, W = W)

def libname_os_specific():
    return "" + select_os_specific(L = "libiomp5.so", M = "libiomp5.dylib", W = "libiomp5md.dll")

# TODO(Intel-tf) Replace the following calls to cc_binary with cc_library.
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

# MacOS build has not been tested, however since the MacOS build of openmp
# uses the same configuration as Linux, the following should work.
def libiomp5_cc_binary(name, cppsources, srcdeps, common_includes):
    cc_binary(
        name = name,
        srcs = cppsources + srcdeps +
               select_os_specific_2(
                   LM = [
                       #linux & macos specific files
                       "runtime/src/z_Linux_util.cpp",
                       "runtime/src/kmp_gsupport.cpp",
                       "runtime/src/z_Linux_asm.S",
                   ],
                   W = [
                       #window specific files
                       "runtime/src/z_Windows_NT_util.cpp",
                       "runtime/src/z_Windows_NT-586_util.cpp",
                       ":openmp_asm",
                   ],
               ),
        copts = select_os_specific_2(
            LM = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
            W = ["/Domp_EXPORTS /D_M_AMD64 /DOMPT_SUPPORT=0 /D_WINDOWS /D_WINNT /D_USRDLL"],
        ),
        includes = common_includes,
        linkopts = select_os_specific_2(
            LM = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
            W = ["/MACHINE:X64"],
        ),
        linkshared = True,
        additional_linker_inputs = select_os_specific_2(
            LM = [":ldscript"],
            W = [":generate_def"],
        ),
        win_def_file = ":generate_def",  # This will be ignored for non Windows builds
        visibility = ["//visibility:public"],
    )
