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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_BASE_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_BASE_H_

#include <cassert>
#include <functional>
#include <vector>

namespace mlir {
namespace disc_ral {

// integer status type, used for error checking
// value zero is always ok, otherwise is failed
using status_t = int32_t;

// memory buffer abstraction
using buffer_t = void*;

// const memory buffer abstraction
using const_buffer_t = const void*;

// opaque resource abstraction
using opaque_t = void*;

// memory buffer shape abstraction
using buffer_shape_t = std::vector<int64_t>;

// ral supported function prototype
using ral_func_t = std::function<void(void**)>;

// Buffer allocation prototype used in ral
using alloc_t = std::function<buffer_t(size_t)>;

// Buffer deallocation prototype used in ral
using dealloc_t = std::function<void(buffer_t)>;

// A class represents different status.
enum ErrorCode {
  kSuccess = 0,
  kUnKnownFailure,
  kInvalidArgument,
  kRuntimeError,
  kNotImplement
};

// Represets a shaped buffer.
struct Tensor {
  buffer_t buffer = nullptr;
  buffer_shape_t shape;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_RUNTIME_RAL_RAL_BASE_H_
