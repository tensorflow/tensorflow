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

#include "tensorflow/python/saved_model/pywrap_saved_model_fingerprinting.h"

#include <exception>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/fingerprinting.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

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

class FileNotFoundException : public std::exception {
 public:
  explicit FileNotFoundException(const char *m) : message_{m} {}
  const char *what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_ = "";
};

void DefineFingerprintingModule(py::module main_module) {
  auto m = main_module.def_submodule("fingerprinting");

  m.doc() = "Python bindings for TensorFlow SavedModel Fingerprinting.";

  static py::exception<FingerprintException> fp_ex(m, "FingerprintException");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const FingerprintException &e) {
      fp_ex(e.what());
    }
  });

  static py::exception<FileNotFoundException> fnf_ex(m,
                                                     "FileNotFoundException");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const FileNotFoundException &e) {
      fnf_ex(e.what());
    }
  });

  m.def(
      "CreateFingerprintDef",
      [](std::string export_dir) -> absl::StatusOr<py::bytes> {
        absl::StatusOr<FingerprintDef> fingerprint =
            fingerprinting::CreateFingerprintDef(export_dir);
        if (fingerprint.ok()) {
          return py::bytes(fingerprint.value().SerializeAsString());
        }
        throw FingerprintException(
            absl::StrCat(
                std::string("Could not create fingerprint in directory: " +
                            export_dir),
                "\n", fingerprint.status().ToString())
                .c_str());
      },
      py::arg("export_dir"),
      py::doc(
          "Returns the serialized FingerprintDef of a SavedModel on disk."));

  m.def(
      "ReadSavedModelFingerprint",
      [](std::string export_dir) {
        absl::StatusOr<FingerprintDef> fingerprint =
            fingerprinting::ReadSavedModelFingerprint(export_dir);
        if (fingerprint.ok()) {
          return py::bytes(fingerprint.value().SerializeAsString());
        } else if (fingerprint.status().code() == absl::StatusCode::kNotFound) {
          throw FileNotFoundException(
              absl::StrCat(
                  std::string("Could not find fingerprint in directory: " +
                              export_dir),
                  "\n", fingerprint.status().ToString())
                  .c_str());
        } else {
          throw FingerprintException(
              absl::StrCat(
                  std::string(
                      "Could not read fingerprint from fingerprint.pb file "
                      "in directory: " +
                      export_dir),
                  "\n", fingerprint.status().ToString())
                  .c_str());
        }
      },
      py::arg("export_dir"),
      py::doc(
          "Loads the `fingerprint.pb` from `export_dir`, returns an error if "
          "there is none."));

  m.def(
      "SingleprintFromFP",
      [](std::string export_dir) {
        absl::StatusOr<std::string> singleprint =
            fingerprinting::Singleprint(export_dir);
        if (singleprint.ok()) {
          return py::str(singleprint.value());
        }
        throw FingerprintException(
            absl::StrCat(
                std::string("Could not create singleprint from the fingerprint "
                            "specified by the export_dir."),
                "\n", singleprint.status().ToString())
                .c_str());
      },
      py::arg("export_dir"),
      py::doc("Canonical fingerprinting ID for a SavedModel."));

  m.def(
      "SingleprintFromSM",
      [](std::string export_dir) {
        absl::StatusOr<FingerprintDef> fingerprint_def =
            fingerprinting::CreateFingerprintDef(export_dir);
        if (!fingerprint_def.ok()) {
          throw FingerprintException(
              absl::StrCat(
                  std::string(
                      "Could not create singleprint from the saved_model."),
                  "\n", fingerprint_def.status().ToString())
                  .c_str());
        }

        absl::StatusOr<std::string> singleprint =
            fingerprinting::Singleprint(fingerprint_def.value());
        if (!singleprint.ok()) {
          throw FingerprintException(
              absl::StrCat(
                  std::string(
                      "Could not create singleprint from the saved_model."),
                  "\n", singleprint.status().ToString())
                  .c_str());
        }

        return py::str(singleprint.value());
      },
      py::arg("export_dir"),
      py::doc("Canonical fingerprinting ID for a SavedModel."));

  m.def(
      "Singleprint",
      [](uint64 graph_def_program_hash, uint64 signature_def_hash,
         uint64 saved_object_graph_hash, uint64 checkpoint_hash) {
        absl::StatusOr<std::string> singleprint = fingerprinting::Singleprint(
            graph_def_program_hash, signature_def_hash, saved_object_graph_hash,
            checkpoint_hash);
        if (singleprint.ok()) {
          return py::str(singleprint.value());
        }
        throw FingerprintException(
            absl::StrCat(
                std::string("Could not create singleprint from given values."),
                "\n", singleprint.status().ToString())
                .c_str());
      },
      py::arg("graph_def_program_hash"), py::arg("signature_def_hash"),
      py::arg("saved_object_graph_hash"), py::arg("checkpoint_hash"),
      py::doc("Canonical fingerprinting ID for a SavedModel."));
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
