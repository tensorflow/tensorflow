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

// Defines the pywrap_libexport module. In order to have only one dynamically-
// linked shared object, all SavedModel C++ APIs must be added here.

#include "pybind11/pybind11.h"
#include "tensorflow/cc/experimental/libexport/constants.h"
#include "tensorflow/cc/experimental/libexport/save.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/saved_model/experimental/pywrap_libexport_metrics.h"

namespace py = pybind11;

PYBIND11_MODULE(pywrap_libexport, m) {
  m.doc() = "TensorFlow exporter Python bindings";

  m.attr("ASSETS_DIRECTORY") = py::str(tensorflow::libexport::kAssetsDirectory);

  m.attr("EXTRA_ASSETS_DIRECTORY") =
      py::str(tensorflow::libexport::kExtraAssetsDirectory);

  m.attr("ASSETS_KEY") = py::str(tensorflow::libexport::kAssetsKey);

  m.attr("DEBUG_DIRECTORY") = py::str(tensorflow::libexport::kDebugDirectory);

  m.attr("DEBUG_INFO_FILENAME_PB") =
      py::str(tensorflow::libexport::kDebugInfoFilenamePb);

  m.attr("INIT_OP_SIGNATURE_KEY") =
      py::str(tensorflow::libexport::kInitOpSignatureKey);

  m.attr("LEGACY_INIT_OP_KEY") =
      py::str(tensorflow::libexport::kLegacyInitOpKey);

  m.attr("MAIN_OP_KEY") = py::str(tensorflow::libexport::kMainOpKey);

  m.attr("TRAIN_OP_KEY") = py::str(tensorflow::libexport::kTrainOpKey);

  m.attr("TRAIN_OP_SIGNATURE_KEY") =
      py::str(tensorflow::libexport::kTrainOpSignatureKey);

  m.attr("SAVED_MODEL_FILENAME_PB") =
      py::str(tensorflow::libexport::kSavedModelFilenamePb);

  m.attr("SAVED_MODEL_FILENAME_PBTXT") =
      py::str(tensorflow::libexport::kSavedModelFilenamePbtxt);

  m.attr("SAVED_MODEL_SCHEMA_VERSION") =
      tensorflow::libexport::kSavedModelSchemaVersion;

  m.attr("VARIABLES_DIRECTORY") =
      py::str(tensorflow::libexport::kVariablesDirectory);

  m.attr("VARIABLES_FILENAME") =
      py::str(tensorflow::libexport::kVariablesFilename);

  m.def("Save", [](const char* export_dir) {
    tensorflow::MaybeRaiseFromStatus(tensorflow::libexport::Save(export_dir));
  });

  tensorflow::DefineMetricsModule(m);
}
