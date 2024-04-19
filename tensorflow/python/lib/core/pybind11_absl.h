/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_ABSL_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_ABSL_H_

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/core/platform/stringpiece.h"

#ifndef ABSL_USES_STD_STRING_VIEW

namespace pybind11 {
namespace detail {

// Convert between tensorflow::StringPiece (aka absl::string_view) and Python.
//
// pybind11 supports std::string_view, and absl::string_view is meant to be a
// drop-in replacement for std::string_view, so we can just use the built in
// implementation.
template <>
struct type_caster<tensorflow::StringPiece>
    : string_caster<tensorflow::StringPiece, true> {};

}  // namespace detail
}  // namespace pybind11

#endif  // ABSL_USES_STD_STRING_VIEW
#endif  // TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_ABSL_H_
