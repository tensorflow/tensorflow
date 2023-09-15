/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tsl/python/lib/core/ml_dtypes.h"

#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11

namespace tsl {
namespace ml_dtypes {

namespace py = pybind11;

const NumpyDtypes& GetNumpyDtypes() {
  static const NumpyDtypes* numpy_dtypes = []() {
    py::module ml_dtypes = py::module::import("ml_dtypes");
    NumpyDtypes* dtypes = new NumpyDtypes();
    dtypes->bfloat16 = py::dtype::from_args(ml_dtypes.attr("bfloat16")).num();
    dtypes->float8_e4m3fn =
        py::dtype::from_args(ml_dtypes.attr("float8_e4m3fn")).num();
    dtypes->float8_e5m2 =
        py::dtype::from_args(ml_dtypes.attr("float8_e5m2")).num();
    dtypes->float8_e4m3b11fnuz =
        py::dtype::from_args(ml_dtypes.attr("float8_e4m3b11fnuz")).num();
    dtypes->float8_e4m3fnuz =
        py::dtype::from_args(ml_dtypes.attr("float8_e4m3fnuz")).num();
    dtypes->float8_e5m2fnuz =
        py::dtype::from_args(ml_dtypes.attr("float8_e5m2fnuz")).num();
    dtypes->int4 = py::dtype::from_args(ml_dtypes.attr("int4")).num();
    dtypes->uint4 = py::dtype::from_args(ml_dtypes.attr("uint4")).num();
    return dtypes;
  }();
  return *numpy_dtypes;
}

}  // namespace ml_dtypes
}  // namespace tsl
