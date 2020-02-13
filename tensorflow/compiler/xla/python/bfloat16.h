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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_BFLOAT16_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_BFLOAT16_H_

#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

xla::StatusOr<pybind11::object> Bfloat16Dtype();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_BFLOAT16_H_
