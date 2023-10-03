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

#include <exception>

#include "numpy/ndarraytypes.h"
#include "absl/base/call_once.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tsl/python/lib/core/numpy.h"  // IWYU pragma: keep

namespace tsl {
namespace ml_dtypes {

namespace py = pybind11;

namespace {

class MlDtypesInitInfo {
 public:
  MlDtypesInitInfo()
      : np_dtypes_{/*bfloat16=*/NPY_NOTYPE,
                   /*float8_e4m3fn=*/NPY_NOTYPE,
                   /*float8_e4m3b11fnuz=*/NPY_NOTYPE,
                   /*float8_e4m3fnuz=*/NPY_NOTYPE,
                   /*float8_e5m2=*/NPY_NOTYPE,
                   /*float8_e5m2fnuz=*/NPY_NOTYPE,
                   /*int4=*/NPY_NOTYPE,
                   /*uint4=*/NPY_NOTYPE},
        valid_{false} {}

  void Init() {
    valid_ = true;
    // Pybind11 might throw.
    try {
      py::gil_scoped_acquire acquire;
      py::module ml_dtypes = py::module::import("ml_dtypes");
      np_dtypes_.bfloat16 =
          py::dtype::from_args(ml_dtypes.attr("bfloat16")).num();
      np_dtypes_.float8_e4m3fn =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3fn")).num();
      np_dtypes_.float8_e5m2 =
          py::dtype::from_args(ml_dtypes.attr("float8_e5m2")).num();
      np_dtypes_.float8_e4m3b11fnuz =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3b11fnuz")).num();
      np_dtypes_.float8_e4m3fnuz =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3fnuz")).num();
      np_dtypes_.float8_e5m2fnuz =
          py::dtype::from_args(ml_dtypes.attr("float8_e5m2fnuz")).num();
      np_dtypes_.int4 = py::dtype::from_args(ml_dtypes.attr("int4")).num();
      np_dtypes_.uint4 = py::dtype::from_args(ml_dtypes.attr("uint4")).num();
    } catch (const std::exception& e) {
      py::print(e.what());
      valid_ = false;
    }

    // Verify all types were successfully loaded.
    if (np_dtypes_.bfloat16 == NPY_NOTYPE ||
        np_dtypes_.float8_e4m3fn == NPY_NOTYPE ||
        np_dtypes_.float8_e4m3fnuz == NPY_NOTYPE ||
        np_dtypes_.float8_e4m3b11fnuz == NPY_NOTYPE ||
        np_dtypes_.float8_e5m2 == NPY_NOTYPE ||
        np_dtypes_.float8_e5m2fnuz == NPY_NOTYPE ||
        np_dtypes_.int4 == NPY_NOTYPE || np_dtypes_.uint4 == NPY_NOTYPE) {
      valid_ = false;
    }
  }

  bool IsValid() const { return valid_; }

  const NumpyDtypes& GetNumpyDtypes() const { return np_dtypes_; }

 private:
  NumpyDtypes np_dtypes_;  // Numpy type numbers.
  bool valid_;             // Stores whether type loading was valid.
};

// Safely initialize the ml_dtypes module and load the numpy dtype information.
const MlDtypesInitInfo& GetMlDtypesInitInfo() {
  static MlDtypesInitInfo info;

  // We must take special care in initializing the ml_dtypes module
  // since there is a potential race condition between the python
  // GIL and any synchronization mechanism we attempt to use (b/302750630).
  // We also want to avoid unnecessarily locking the GIL if possible.
  static bool initialized = false;
  if (!initialized) {
    auto init = [&]() { info.Init(); };

    // GIL must be released prior to attempting synchronization.
    py::gil_scoped_release release;
    static absl::once_flag init_flag;
    absl::call_once(init_flag, init);
    initialized = true;
  }

  return info;
}

}  // namespace

const NumpyDtypes& GetNumpyDtypes() {
  return GetMlDtypesInitInfo().GetNumpyDtypes();
}

bool RegisterTypes() { return GetMlDtypesInitInfo().IsValid(); }

}  // namespace ml_dtypes
}  // namespace tsl
