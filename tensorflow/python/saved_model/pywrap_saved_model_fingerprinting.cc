/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "tensorflow/cc/saved_model/fingerprinting.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
namespace saved_model {
namespace python {

namespace py = pybind11;

void DefineFingerprintingModule(py::module main_module) {
  auto m = main_module.def_submodule("fingerprinting");

  m.doc() = "Python bindings for TensorFlow SavedModel Fingerprinting.";

  m.def(
      "CreateFingerprintDef",
      [](std::string serialized_saved_model, std::string export_dir) {
        // Deserialize the SavedModel.
        SavedModel saved_model_pb;
        saved_model_pb.ParseFromString(serialized_saved_model);

        return py::bytes(
            fingerprinting::CreateFingerprintDef(saved_model_pb, export_dir)
                .SerializeAsString());
      },
      py::arg("saved_model"), py::arg("export_dir"),
      py::doc(
          "Returns the serialized FingerprintDef of a serialized SavedModel."));

  m.def(
      "MaybeReadSavedModelChecksum",
      [](std::string export_dir) {
        StatusOr<FingerprintDef> fingerprint =
            fingerprinting::ReadSavedModelFingerprint(export_dir);
        if (fingerprint.ok()) {
          return fingerprint->saved_model_checksum();
        }
        return (uint64_t)0;
      },
      py::arg("export_dir"),
      py::doc(
          "Reads the fingerprint checksum from SavedModel directory. Returns "
          "0 if an error occurs."));
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
