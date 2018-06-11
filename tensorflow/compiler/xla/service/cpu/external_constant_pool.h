/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_EXTERNAL_CONSTANT_POOL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_EXTERNAL_CONSTANT_POOL_H_

#include <memory>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/mem.h"

namespace xla {
namespace cpu {
// An ExternalConstantPool maintains a set of constants kept external to
// generated LLVM IR. These constants are accessed from the IR via globals with
// extern linkage.  This current incarnation of ExternalConstantPool only
// supports the JIT CPU backend; the AOT backend is not supported.
//
// Implementation-wise, this is a simple wrapper around a map of strings to byte
// buffers.  This simply implementation works in a JIT scenario.  This class
// will have to become smarter if we decide to support external constant pools
// on AOT compiles in the future.
class ExternalConstantPool {
 public:
  // Inserts a buffer with the contents of `literal` into the constant pool with
  // the name `name`.  It is an error to try to insert two constants with the
  // same `name` into the same constant pool.  The buffer for literal is aligned
  // to `aligment` bytes, and `alignment` must be a power of 2.
  //
  // The constant pool copies out the contents of `literal` into a buffer it
  // owns -- it does not keep pointers to `literal`, or to memory owned by
  // `literal`.
  void Insert(string name, const LiteralSlice& literal, int64 alignment);

  // Find the constant with name `name` in this constant pool.  If there isn't
  // such constant, return nullptr.
  const uint8* Find(const string& name);

 private:
  // We need to `AlignedFree` pointers allocated into `entries_` since we
  // allocate them with `AlignedMalloc`.
  struct FreeDeleter {
    void operator()(void* ptr) { tensorflow::port::AlignedFree(ptr); }
  };

  tensorflow::gtl::FlatMap<string, std::unique_ptr<uint8, FreeDeleter>>
      entries_;
};
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_EXTERNAL_CONSTANT_POOL_H_
