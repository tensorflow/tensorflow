"""Bazel build definitions for oneDNN GPU support.

This module provides macros to embed OpenCL kernels and header files as C++
sources for oneDNN GPU backend builds with SYCL. These are used when building
oneDNN with GPU acceleration support.
"""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def convert_cl_to_cpp(name, src, cl_list, **kwargs):
    """Shorthand to generate embedded C++ sources from OpenCL kernels.

    Args:
      name: name of the genrule target.
      src: template input file used to generate the kernel list.
      cl_list: list of .cl kernel source files to embed.
      **kwargs: additional arguments forwarded to native.genrule.

    Returns:
      None. This macro emits a genrule that produces C++ sources.
    """
    cpp_list = [cl.replace(".cl", "_kernel.cpp") for cl in cl_list]
    kernel_list = src.replace(".in", "")
    cpp_list.append(kernel_list)

    tool = "@xla//third_party/mkl_dnn:gen_gpu_kernel_list"

    native.genrule(
        name = name,
        srcs = [src] + cl_list,
        outs = cpp_list,
        tools = [tool],
        cmd = "$(location {}) --in=$(location {}) --out=$(RULEDIR) --header=False".format(tool, src),
        **kwargs
    )

def convert_header_to_cpp(name, src, header_list, **kwargs):
    """Shorthand to generate embedded C++ sources from header files.

    Args:
      name: name of the genrule target.
      src: template input file used to generate the header list.
      header_list: list of .h header source files to embed.
      **kwargs: additional arguments forwarded to native.genrule.

    Returns:
      None. This macro emits a genrule that produces C++ sources.
    """
    cpp_list = []
    h_list = []
    for h in header_list:
        if h.endswith(".h"):
            h_list.append(h.replace(".h", "_header.cpp"))
    cpp_list.extend(h_list)

    tool = "@xla//third_party/mkl_dnn:gen_gpu_kernel_list"

    native.genrule(
        name = name,
        srcs = [src] + header_list,
        outs = cpp_list,
        tools = [tool],
        cmd = "$(location {}) --in=$(location {}) --out=$(RULEDIR) --header=True".format(tool, src),
        **kwargs
    )
