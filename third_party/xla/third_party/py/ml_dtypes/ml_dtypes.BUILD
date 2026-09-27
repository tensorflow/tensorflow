# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Main ml_dtypes library. """

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "float8",
    hdrs = ["include/float8.h"],
    deps = ["@eigen_archive//:eigen3"],
)

cc_library(
    name = "intn",
    hdrs = ["include/intn.h"],
)

cc_library(
    name = "mxfloat",
    hdrs = ["include/mxfloat.h"],
    deps = [
        ":float8",
        "@eigen_archive//:eigen3",
    ],
)

pybind_extension(
    name = "_ml_dtypes_ext",
    srcs = [
        "_src/common.h",
        "_src/custom_float.h",
        "_src/dtypes.cc",
        "_src/intn_numpy.h",
        "_src/numpy.cc",
        "_src/numpy.h",
        "_src/ufuncs.h",
    ],
    visibility = [":__subpackages__"],
    deps = [
        ":float8",
        ":intn",
        ":mxfloat",
        "@eigen_archive//:eigen3",
        "@xla//third_party/py/numpy:headers",
    ],
)

py_library(
    name = "ml_dtypes",
    srcs = [
        "__init__.py",
        "_finfo.py",
        "_iinfo.py",
    ],
    data = [":_ml_dtypes_ext"],
    imports = ["."],  # Import relative to _this_ directory, not the root.
)
