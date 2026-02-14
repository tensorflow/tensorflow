load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

# Intel(R) Software Development Tools Licensed under the Intel End User License Agreement for Developer Tools (Version August 2024)
# Tools -> Intel(R) oneAPI DPC++/C++ Compiler, Intel(R) Vtune(TM) Profiler
# Intel(R) Software Development Tools Licensed under the Intel Simplified Software License (Version October 2022)  
# Tools -> oneAPI Math Kernel Library (oneMKL)
# Intel(R) Software Development Tools Licensed under Open Source Licenses Apache License, Version 2.0 
# Tools -> oneAPI Deep Neural Network Library, Intel(R) oneAPI Data Analytics Library (oneDAL)
# Apache License, Version 2.0 with LLVM Exception -- Tools ->Intel(R) oneAPI DPC++/C++ Compiler,Intel(R) oneAPI DPC++ Library (oneDPL)
# The GNU General Public License v3.0 -> Tools-- Intel(R) Distribution for GDB*
licenses(["restricted"])  

config_setting(
    name = "using_sycl",
    values = {
        "define": "using_sycl=true",
    },
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sycl_headers",
    hdrs = [
        %{sycl_headers}
    ],
    includes = [
        ".",
        "sycl/include",
        "sycl/include/sycl",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "level_zero",
    srcs = [
        %{level_zero_libs}
    ],
    data = [
        %{level_zero_libs}
    ],
    hdrs = [
        %{level_zero_headers}
    ],
    includes = [
        ".",
        "level_zero/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl",
    srcs = [
        %{mkl_intel_ilp64_src}
        %{mkl_sequential_src}
        %{mkl_core_src}
        %{mkl_sycl_srcs}
    ],
    data = [
        %{mkl_intel_ilp64_src}
        %{mkl_sequential_src}
        %{mkl_core_src}
        %{mkl_sycl_srcs}

    ],
    includes = [
        ".",
        "sycl/include",
    ],
    linkopts = ["-Wl,-Bstatic,-lsvml,-lirng,-limf,-lirc,-lirc_s,-Bdynamic"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

%{copy_rules}
