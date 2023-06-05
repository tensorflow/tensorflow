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

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/cc/saved_model/constants.h"

namespace tensorflow {
namespace saved_model {
namespace python {

namespace py = pybind11;

void DefineConstantsModule(py::module main_module) {
  auto m = main_module.def_submodule("constants");

  m.doc() = "Python bindings for TensorFlow SavedModel Constants";

  m.attr("ASSETS_DIRECTORY") = py::str(tensorflow::kSavedModelAssetsDirectory);

  m.attr("EXTRA_ASSETS_DIRECTORY") =
      py::str(tensorflow::kSavedModelAssetsExtraDirectory);

  m.attr("ASSETS_KEY") = py::str(tensorflow::kSavedModelAssetsKey);

  m.attr("DEBUG_DIRECTORY") = py::str(tensorflow::kSavedModelDebugDirectory);

  m.attr("DEBUG_INFO_FILENAME_PB") =
      py::str(tensorflow::kSavedModelDebugInfoFilenamePb);

  m.attr("INIT_OP_SIGNATURE_KEY") =
      py::str(tensorflow::kSavedModelInitOpSignatureKey);

  m.attr("LEGACY_INIT_OP_KEY") =
      py::str(tensorflow::kSavedModelLegacyInitOpKey);

  m.attr("MAIN_OP_KEY") = py::str(tensorflow::kSavedModelMainOpKey);

  m.attr("TRAIN_OP_KEY") = py::str(tensorflow::kSavedModelTrainOpKey);

  m.attr("TRAIN_OP_SIGNATURE_KEY") =
      py::str(tensorflow::kSavedModelTrainOpSignatureKey);

  m.attr("SAVED_MODEL_FILENAME_PB") =
      py::str(tensorflow::kSavedModelFilenamePb);

  m.attr("SAVED_MODEL_FILENAME_PBTXT") =
      py::str(tensorflow::kSavedModelFilenamePbTxt);

  m.attr("SAVED_MODEL_SCHEMA_VERSION") = tensorflow::kSavedModelSchemaVersion;

  m.attr("VARIABLES_DIRECTORY") =
      py::str(tensorflow::kSavedModelVariablesDirectory);

  m.attr("VARIABLES_FILENAME") =
      py::str(tensorflow::kSavedModelVariablesFilename);

  m.attr("FINGERPRINT_FILENAME") = py::str(tensorflow::kFingerprintFilenamePb);
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
