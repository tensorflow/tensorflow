# Description:
#   opt_einsum is a library for optimizing tensor contraction order for einsum.

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "opt_einsum",
    srcs = [
        "opt_einsum/__init__.py",
        "opt_einsum/_version.py",
        "opt_einsum/backends/__init__.py",
        "opt_einsum/backends/cupy.py",
        "opt_einsum/backends/dispatch.py",
        "opt_einsum/backends/tensorflow.py",
        "opt_einsum/backends/theano.py",
        "opt_einsum/backends/torch.py",
        "opt_einsum/blas.py",
        "opt_einsum/compat.py",
        "opt_einsum/contract.py",
        "opt_einsum/helpers.py",
        "opt_einsum/parser.py",
        "opt_einsum/paths.py",
        "opt_einsum/sharing.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)
