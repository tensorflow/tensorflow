"""This file contains BUILD extensions for building llvm_openmp.
TODO(Intel-tf): Delete this and reuse a similar function in third_party/llvm
after the TF 2.4 branch cut has passed.
"""

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
)

WINDOWS_MSVC_LLVM_OPENMP_LIBPATH = "bazel-out/x64_windows-opt/bin/external/llvm_openmp/libiomp5.dll.if.lib"
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
    return "libiomp5" + select_os_specific(L = ".so", M = ".dylib", W = ".dll")
