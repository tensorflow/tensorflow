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
#include "tensorflow/tsl/python/lib/core/float8.h"

PYBIND11_MODULE(_pywrap_float8, m) {
  tsl::RegisterNumpyFloat8e4m3fn();
  tsl::RegisterNumpyFloat8e5m2();

  m.def("TF_float8_e4m3fn_type",
        [] { return pybind11::handle(tsl::Float8e4m3fnDtype()); });

  m.def("TF_float8_e5m2_type",
        [] { return pybind11::handle(tsl::Float8e5m2Dtype()); });
}
