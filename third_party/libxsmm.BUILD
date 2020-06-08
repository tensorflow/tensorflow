# Description:
#    LIBXSMM: Library for small matrix-matrix multiplications targeting Intel Architecture (x86).

licenses(["notice"])  # BSD 3-clause

exports_files(["LICENSE.md"])

# Arguments to ./scripts/libxsmm_interface.py, see that file for detailed description.
#  precision: SP & DP
#  prefetch: 1 (auto)
libxsmm_interface_arguments = "0 1"

# Arguments to ./scripts/libxsmm_config.py, see that file for detailed description.
# rely on default arguments
libxsmm_config_arguments = ""

# Arguments to ./scripts/libxsmm_dispatch.py, see that file for detailed description.
#  (dummy argument)
libxsmm_dispatch_arguments = "0"

genrule(
    name = "libxsmm_headers",
    srcs = [
        "src/template/libxsmm.h",
        "src/template/libxsmm_config.h",
    ],
    outs = [
        "include/libxsmm.h",
        "include/libxsmm_config.h",
        "include/libxsmm_dispatch.h",
    ],
    cmd = "$(location :libxsmm_interface) $(location src/template/libxsmm.h) " + libxsmm_interface_arguments + " > $(location include/libxsmm.h);" +
          "$(location :libxsmm_config) $(location src/template/libxsmm_config.h) " + libxsmm_config_arguments + " > $(location include/libxsmm_config.h);" +
          "$(location :libxsmm_dispatch) " + libxsmm_dispatch_arguments + " > $(location include/libxsmm_dispatch.h)",
    tools = [
        ":libxsmm_config",
        ":libxsmm_dispatch",
        ":libxsmm_interface",
    ],
    visibility = [
        "//tensorflow/core/kernels:__pkg__",
        "//third_party/eigen3:__pkg__",
    ],
)

cc_library(
    name = "xsmm_avx",
    srcs = glob(
        [
            # general source files (translation units)
            "src/generator_*.c",
            "src/libxsmm_*.c",
        ],
        exclude = [
            # exclude generators (with main functions)
            "src/libxsmm_generator_*.c",
        ],
    ),
    hdrs = glob(
        [
            # general header files
            "include/libxsmm_*.h",
            # trigger rebuild if template changed
            "src/template/*.c",
        ],
        exclude = [
            # exclude existing/generated headers
            "include/libxsmm.h",
            "include/libxsmm_config.h",
            "include/libxsmm_dispatch.h",
        ],
    ) + [
        # source files included internally
        "src/libxsmm_hash.c",
        # generated header files
        "include/libxsmm.h",
        "include/libxsmm_config.h",
        "include/libxsmm_dispatch.h",
    ],
    #copts = [
    #    "-mavx",  # JIT does not work without avx anyway, and this silences some CRC32 warnings.
    #    "-Wno-vla",  # Libxsmm convolutions heavily use VLA.
    #],
    defines = [
        "LIBXSMM_BUILD",
        "LIBXSMM_CTOR",
        "__BLAS=0",
    ],
    includes = [
        "include",
        "src",
        "src/template",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "libxsmm_scripts",
    srcs = glob(["scripts/*.py"]),
    data = ["version.txt"],
)

py_binary(
    name = "libxsmm_interface",
    srcs = ["scripts/libxsmm_interface.py"],
    deps = [":libxsmm_scripts"],
)

py_binary(
    name = "libxsmm_config",
    srcs = ["scripts/libxsmm_config.py"],
    deps = [":libxsmm_scripts"],
)

py_binary(
    name = "libxsmm_dispatch",
    srcs = ["scripts/libxsmm_dispatch.py"],
    deps = [":libxsmm_scripts"],
)
