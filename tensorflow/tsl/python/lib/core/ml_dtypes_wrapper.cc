/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/tsl/python/lib/core/ml_dtypes.h"

PYBIND11_MODULE(pywrap_ml_dtypes, m) {
  tsl::ml_dtypes::RegisterTypes();
  m.def("bfloat16",
        [] { return pybind11::handle(tsl::ml_dtypes::GetBfloat16Dtype()); });
  m.def("float8_e4m3b11fnuz", [] {
    return pybind11::handle(tsl::ml_dtypes::GetFloat8E4m3b11fnuzDtype());
  });
  m.def("float8_e4m3fn", [] {
    return pybind11::handle(tsl::ml_dtypes::GetFloat8E4m3fnDtype());
  });
  m.def("float8_e5m2",
        [] { return pybind11::handle(tsl::ml_dtypes::GetFloat8E5m2Dtype()); });
}
