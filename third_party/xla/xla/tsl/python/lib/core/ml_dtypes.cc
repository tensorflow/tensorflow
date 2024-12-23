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
#include "xla/tsl/python/lib/core/ml_dtypes.h"

#include <atomic>
#include <exception>

// Must be included first to ensure `NPY_NO_DEPRECATED_API` is defined.
// clang-format off
#include "xla/tsl/python/lib/core/numpy.h"  // IWYU pragma: keep
// clang-format on
#include "numpy/ndarraytypes.h"
#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "pybind11/gil.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace tsl {
namespace ml_dtypes {
namespace {

namespace py = pybind11;

struct MlDtypesInitInfo {
  constexpr MlDtypesInitInfo()
      : numpy_dtypes{}, init_valid(true), init_done(false), init_once() {}

  void InitOnce() {
    // Fast/slow path: only interact with GIL if we have not already
    // initialized.
    // IMPORTANT: The lock order matters, so as to avoid deadlock:
    //   First: lock the once-mutex.
    //   Second: lock the GIL.
    //
    // This initialization can happen from within a Python call, when the GIL is
    // already held, in which case we release it first.

    if (!init_done.load(std::memory_order_acquire)) {
      py::gil_scoped_release release;
      absl::call_once(init_once, [this]() { DoInit(); });
    }
  }

  void DoInit() {
    try {
      py::gil_scoped_acquire acquire;
      py::module ml_dtypes = py::module::import("ml_dtypes");

      numpy_dtypes.bfloat16 =
          py::dtype::from_args(ml_dtypes.attr("bfloat16")).num();
      numpy_dtypes.float8_e3m4 =
          py::dtype::from_args(ml_dtypes.attr("float8_e3m4")).num();
      numpy_dtypes.float8_e4m3 =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3")).num();
      numpy_dtypes.float8_e4m3fn =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3fn")).num();
      numpy_dtypes.float8_e5m2 =
          py::dtype::from_args(ml_dtypes.attr("float8_e5m2")).num();
      numpy_dtypes.float8_e4m3b11fnuz =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3b11fnuz")).num();
      numpy_dtypes.float8_e4m3fnuz =
          py::dtype::from_args(ml_dtypes.attr("float8_e4m3fnuz")).num();
      numpy_dtypes.float8_e5m2fnuz =
          py::dtype::from_args(ml_dtypes.attr("float8_e5m2fnuz")).num();
      numpy_dtypes.int4 = py::dtype::from_args(ml_dtypes.attr("int4")).num();
      numpy_dtypes.uint4 = py::dtype::from_args(ml_dtypes.attr("uint4")).num();
    } catch (const std::exception& e) {
      py::gil_scoped_acquire acquire;
      py::print(e.what());
      init_valid = false;
    }

    // Verify all types were successfully loaded.
    if (numpy_dtypes.bfloat16 == NPY_NOTYPE ||
        numpy_dtypes.float8_e3m4 == NPY_NOTYPE ||
        numpy_dtypes.float8_e4m3 == NPY_NOTYPE ||
        numpy_dtypes.float8_e4m3fn == NPY_NOTYPE ||
        numpy_dtypes.float8_e4m3fnuz == NPY_NOTYPE ||
        numpy_dtypes.float8_e4m3b11fnuz == NPY_NOTYPE ||
        numpy_dtypes.float8_e5m2 == NPY_NOTYPE ||
        numpy_dtypes.float8_e5m2fnuz == NPY_NOTYPE ||
        numpy_dtypes.int4 == NPY_NOTYPE || numpy_dtypes.uint4 == NPY_NOTYPE) {
      init_valid = false;
    }

    init_done.store(true, std::memory_order_release);
  }

  NumpyDtypes numpy_dtypes;

  bool init_valid;
  std::atomic<bool> init_done;
  absl::once_flag init_once;
};

const MlDtypesInitInfo& GetMlDtypesInitInfo() {
  ABSL_CONST_INIT static MlDtypesInitInfo state;
  state.InitOnce();
  return state;
}

}  // namespace

const NumpyDtypes& GetNumpyDtypes() {
  return GetMlDtypesInitInfo().numpy_dtypes;
}

bool RegisterTypes() {  //
  return GetMlDtypesInitInfo().init_valid;
}

}  // namespace ml_dtypes
}  // namespace tsl
