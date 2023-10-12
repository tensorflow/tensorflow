/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Defines the pywrap_saved_model module. In order to have only one dynamically-
// linked shared object, all SavedModel python bindings should be added here.

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/cc/experimental/libexport/save.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/saved_model/pywrap_saved_model_constants.h"
#include "tensorflow/python/saved_model/pywrap_saved_model_fingerprinting.h"
// Placeholder for protosplitter merger include.
#include "tensorflow/python/saved_model/pywrap_saved_model_metrics.h"

namespace tensorflow {
namespace saved_model {
namespace python {

PYBIND11_MODULE(pywrap_saved_model, m) {
  m.doc() = "TensorFlow SavedModel Python bindings";

  m.def("Save", [](const char* export_dir) {
    MaybeRaiseFromStatus(libexport::Save(export_dir));
  });

  DefineConstantsModule(m);
  DefineMetricsModule(m);
  DefineFingerprintingModule(m);
  // Placeholder for protosplitter merger module definition.
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
