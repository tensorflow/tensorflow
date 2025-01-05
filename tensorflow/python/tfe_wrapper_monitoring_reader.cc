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
/* 
tfe_wrapper_monitoring_reader.cc
================================
This file provides a Python-C++ binding for TensorFlow Eager's Monitoring Counter API using Pybind11. It bridges TensorFlow's internal monitoring counters (C++) and their usage in Python, enabling developers to analyze runtime performance and diagnose issues.
==============================================================================*/

#include <memory>
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

// Make the TFE_MonitoringCounterReader class opaque to Python
// Ensures encapsulation of internal TensorFlow components
PYBIND11_MAKE_OPAQUE(TFE_MonitoringCounterReader);

PYBIND11_MODULE(_pywrap_tfe_monitoring_reader, m) {
  /* 
  Python Module `_pywrap_tfe_monitoring_reader`
  =============================================
  Provides Python bindings for TensorFlow Eager Monitoring Counters.
  Exposes the TFE_MonitoringCounterReader class and associated functions.
  */

  // Expose TFE_MonitoringCounterReader class to Python
  py::class_<TFE_MonitoringCounterReader> TFE_MonitoringCounterReader_class(
      m, "TFE_MonitoringCounterReader");

  // Function: TFE_MonitoringNewCounterReader
  // Creates a new counter reader for the given counter name.
  // Arguments:
  //   - name (const char*): Name of the monitoring counter.
  // Returns:
  //   - Pointer to a new TFE_MonitoringCounterReader instance.
  m.def("TFE_MonitoringNewCounterReader", [](const char* name) {
    auto output = TFE_MonitoringNewCounterReader(name);
    return output;
  });

  // Function: TFE_MonitoringReadCounter0
  // Reads the value of a counter without associated labels.
  // Arguments:
  //   - cell_reader (TFE_MonitoringCounterReader*): Instance of the counter reader.
  // Returns:
  //   - Integer value of the counter.
  m.def("TFE_MonitoringReadCounter0",
        [](TFE_MonitoringCounterReader* cell_reader) {
          auto output = TFE_MonitoringReadCounter0(cell_reader);
          return output;
        });

  // Function: TFE_MonitoringReadCounter1
  // Reads the value of a counter for a specific label.
  // Arguments:
  //   - cell_reader (TFE_MonitoringCounterReader*): Instance of the counter reader.
  //   - label (const char*): Label associated with the counter.
  // Returns:
  //   - Integer value of the counter for the given label.
  m.def("TFE_MonitoringReadCounter1",
        [](TFE_MonitoringCounterReader* cell_reader, const char* label) {
          auto output = TFE_MonitoringReadCounter1(cell_reader, label);
          return output;
        });
}
