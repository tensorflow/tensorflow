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

#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/experimental/support/codegen/android_java_generator.h"
#include "tensorflow/lite/experimental/support/codegen/code_generator.h"

namespace tflite {
namespace support {
namespace codegen {

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(_pywrap_codegen, m) {
  pybind11::class_<AndroidJavaGenerator>(m, "AndroidJavaGenerator")
      .def(pybind11::init<const std::string &>())
      .def("generate",
           overload_cast_<const char *, const std::string &,
                          const std::string &, const std::string &>()(
               &AndroidJavaGenerator::Generate))
      .def("get_error_message", &AndroidJavaGenerator::GetErrorMessage);
  pybind11::class_<GenerationResult>(m, "GenerationResult")
      .def(pybind11::init<>())
      .def_readwrite("files", &GenerationResult::files);
  pybind11::class_<GenerationResult::File>(m, "GenerationResultFile")
      .def(pybind11::init<>())
      .def_readwrite("path", &GenerationResult::File::path)
      .def_readwrite("content", &GenerationResult::File::content);
}

}  // namespace codegen
}  // namespace support
}  // namespace tflite
