# Description:
#   Implib.so is a simple equivalent of Windows DLL import libraries for POSIX
#   shared libraries.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

exports_files([
    "LICENSE.txt",
])

py_library(
    name = "implib_gen_lib",
    srcs = ["implib-gen.py"],
    data = glob([
        "arch/**/*.S.tpl",
        "arch/**/*.ini",
    ]),
)
