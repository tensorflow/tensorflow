/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_RUNTIME_FFI_FFI_ABI_H_
#define XLA_RUNTIME_FFI_FFI_ABI_H_

#include <cstdint>

namespace xla {
namespace runtime {
namespace internal {

//===----------------------------------------------------------------------===//
// C structures that XLA uses internally to encode arguments and attributes.
//===----------------------------------------------------------------------===//

// When XLA compiles host-side executable via lowering to LLVM (see `rt-to-llvm`
// pass) it encodes arguments and attributes as `!llvm.struct<...>` types stored
// as LLVM global constants (attributes and statically known arguments) or as
// allocas on the stack. We rely on standard layout C++ structs to reinterpret
// cast arguments and attributes pointers, and convert them to user-friendly C++
// types (e.g. `EncodedMemref` to `StridedBufferArg`).
//
// See: https://en.cppreference.com/w/cpp/types/is_standard_layout

struct EncodedMemref {
  uint8_t dtype;
  uint8_t rank;
  void* data;
  int64_t dims[];
};

template <typename T>
struct EncodedArray {
  int64_t size;
  const T* data;
};

template <typename T>
struct EncodedDenseElements {
  struct EncodedArray<T> payload;
  int64_t rank;
  int64_t shape[];
};

}  // namespace internal
}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_FFI_FFI_ABI_H_
