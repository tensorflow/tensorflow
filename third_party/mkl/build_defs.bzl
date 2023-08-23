"""Starlark macros for MKL.

if_mkl is a conditional to check if we are building with MKL.
if_mkl_ml is a conditional to check if we are building with MKL-ML.
if_mkl_ml_only is a conditional to check for MKL-ML-only (no MKL-DNN) mode.
if_mkl_lnx_x64 is a conditional to check for MKL
if_enable_mkl is a conditional to check if building with MKL and MKL is enabled.

mkl_repository is a repository rule for creating MKL repository rule that can
be pointed to either a local folder, or downloaded from the internet.
mkl_repository depends on the following environment variables:
  * `TF_MKL_ROOT`: The root folder where a copy of libmkl is located.
"""

load(
    "//tensorflow/tsl/mkl:build_defs.bzl",
    _if_enable_mkl = "if_enable_mkl",
    _if_mkl = "if_mkl",
    _if_mkl_lnx_x64 = "if_mkl_lnx_x64",
    _if_mkl_ml = "if_mkl_ml",
    _mkl_deps = "mkl_deps",
    _mkl_repository = "mkl_repository",
)

if_mkl = _if_mkl
if_mkl_ml = _if_mkl_ml
if_mkl_lnx_x64 = _if_mkl_lnx_x64
if_enable_mkl = _if_enable_mkl
mkl_deps = _mkl_deps
mkl_repository = _mkl_repository
