/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PMAP_LIB_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PMAP_LIB_H_

#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/sharded_device_array.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/core/platform/logging.h"

// TODO(jblespiau): The current implementation moves the Python logic to C++,
// as a preliminary step to executing the `pmap` execution path from C++.
// It implements the current Python behavior (thus, it may not be optimal, and
// we will be able to modify it later).

namespace jax {

void BuildPmapSubmodule(pybind11::module& m);

}  // namespace jax

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PMAP_LIB_H_
