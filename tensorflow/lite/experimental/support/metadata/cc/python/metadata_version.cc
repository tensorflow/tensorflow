/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/support/metadata/cc/metadata_version.h"

#include "pybind11/pybind11.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace metadata {

PYBIND11_MODULE(_pywrap_metadata_version, m) {
  m.doc() = R"pbdoc(
    _pywrap_metadata_version
    A module that returns the minimum metadata parser version of a given
    metadata flatbuffer.
  )pbdoc";

  // Using pybind11 type conversions to convert between Python and native
  // C++ types. There are other options to provide access to native Python types
  // in C++ and vice versa. See the pybind 11 instrcution [1] for more details.
  // Type converstions is recommended by pybind11, though the main downside
  // is that a copy of the data must be made on every Python to C++ transition:
  // this is needed since the C++ and Python versions of the same type generally
  // won’t have the same memory layout.
  //
  // [1]: https://pybind11.readthedocs.io/en/stable/advanced/cast/index.html
  m.def("GetMinimumMetadataParserVersion",
        [](const std::string& buffer_data) -> std::string {
          std::string min_version;
          if (GetMinimumMetadataParserVersion(
                  reinterpret_cast<const uint8_t*>(buffer_data.c_str()),
                  buffer_data.length(), &min_version) != kTfLiteOk) {
            pybind11::value_error(
                "Error occurred when getting the minimum metadata parser "
                "version of the metadata flatbuffer.");
          }
          return min_version;
        });
}

}  // namespace metadata
}  // namespace tflite
