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

#include <exception>
#include <string>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/cc/saved_model/fingerprinting.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
namespace saved_model {
namespace python {

namespace py = pybind11;

class FingerprintException : public std::exception {
 public:
  explicit FingerprintException(const char *m) : message_{m} {}
  const char *what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_ = "";
};

void DefineFingerprintingModule(py::module main_module) {
  auto m = main_module.def_submodule("fingerprinting");

  m.doc() = "Python bindings for TensorFlow SavedModel Fingerprinting.";

  static py::exception<FingerprintException> ex(m, "FingerprintException");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const FingerprintException &e) {
      ex(e.what());
    }
  });

  m.def(
      "CreateFingerprintDef",
      [](std::string serialized_saved_model, std::string export_dir) {
        // Deserialize the SavedModel.
        SavedModel saved_model_pb;
        saved_model_pb.ParseFromString(serialized_saved_model);

        StatusOr<FingerprintDef> fingerprint =
            fingerprinting::CreateFingerprintDef(saved_model_pb, export_dir);
        if (fingerprint.ok()) {
          return py::bytes(fingerprint.value().SerializeAsString());
        }
        throw FingerprintException(
            std::string("Could not create fingerprint in directory: " +
                        export_dir)
                .c_str());
      },
      py::arg("saved_model"), py::arg("export_dir"),
      py::doc(
          "Returns the serialized FingerprintDef of a serialized SavedModel."));

  m.def(
      "ReadSavedModelFingerprint",
      [](std::string export_dir) {
        StatusOr<FingerprintDef> fingerprint =
            fingerprinting::ReadSavedModelFingerprint(export_dir);
        if (fingerprint.ok()) {
          return py::bytes(fingerprint.value().SerializeAsString());
        }
        throw FingerprintException(
            std::string("Could not read fingerprint from directory: " +
                        export_dir)
                .c_str());
      },
      py::arg("export_dir"),
      py::doc(
          "Loads the `fingerprint.pb` from `export_dir`, returns an error if "
          "there is none."));

  m.def(
      "Singleprint",
      [](uint64 graph_def_program_hash, uint64 signature_def_hash,
         uint64 saved_object_graph_hash, uint64 checkpoint_hash) {
        StatusOr<std::string> singleprint = fingerprinting::Singleprint(
            graph_def_program_hash, signature_def_hash, saved_object_graph_hash,
            checkpoint_hash);
        if (singleprint.ok()) {
          return py::str(singleprint.value());
        }
        throw FingerprintException(
            std::string("Could not create singleprint from given values.")
                .c_str());
      },
      py::arg("graph_def_program_hash"), py::arg("signature_def_hash"),
      py::arg("saved_object_graph_hash"), py::arg("checkpoint_hash"),
      py::doc("Canonical fingerprinting ID for a SavedModel."));
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
