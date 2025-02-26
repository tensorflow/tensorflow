/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "Python.h"
#include "pybind11/complex.h"  // from @pybind11
#include "pybind11/functional.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/c/eager/c_api_experimental_reader.h"
#include "tensorflow/c/eager/tfe_monitoring_reader_internal.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(TFE_MonitoringCounterReader);

PYBIND11_MODULE(_pywrap_tfe_monitoring_reader, m) {
  py::class_<TFE_MonitoringCounterReader> TFE_MonitoringCounterReader_class(
      m, "TFE_MonitoringCounterReader");
  m.def("TFE_MonitoringNewCounterReader", [](const char* name) {
    auto output = TFE_MonitoringNewCounterReader(name);
    return output;
  });
  m.def("TFE_MonitoringReadCounter0",
        [](TFE_MonitoringCounterReader* cell_reader) {
          auto output = TFE_MonitoringReadCounter0(cell_reader);
          return output;
        });
  m.def("TFE_MonitoringReadCounter1",
        [](TFE_MonitoringCounterReader* cell_reader, const char* label) {
          auto output = TFE_MonitoringReadCounter1(cell_reader, label);
          return output;
        });
};
