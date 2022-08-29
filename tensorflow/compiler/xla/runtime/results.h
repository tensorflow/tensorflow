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

#ifndef XLA_RUNTIME_RESULTS_H_
#define XLA_RUNTIME_RESULTS_H_

#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/types.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Conversions from XLA executable results to C++ types.
//===----------------------------------------------------------------------===//

// The result type defines its own ABI as a required number of bytes and
// alignment, and executable returns results by writing into the requested
// memory allocated in the call frame. The user is responsible for providing
// a conversion function that converts this opaque memory back to the C++
// data type. For example memrefs returned as a `StridedMemrefType` structure,
// and it is the user responsibiity to define a conversion function that can
// convert a memref to the run time Tensor/Buffer type.
//
// It is important that the type that is written into the call frame memory has
// a standard memory layout, because we rely on `reinterpret_cast` to reinterpet
// the opaque bytes to a C struct.
//
// See https://en.cppreference.com/w/cpp/types/is_standard_layout

// Result converter is responsible for taking a pointer to the memory location
// where the executable wrote the result, and converting it to the corresponding
// run time value expected by the caller (e.g. memref descriptor to Tensor).
class ResultConverter {
 public:
  virtual ~ResultConverter() = default;

  // Converts value `ret` of type `runtime_type` (runtime type derived from the
  // original `type`) returned from the executable at `result_index` result
  // position using registered conversion functions. Returns a logical result
  // telling if the conversion was successful.
  virtual LogicalResult ReturnValue(unsigned result_index, const Type* type,
                                    const Type* runtime_type,
                                    void* ret) const = 0;

  // Returns error for all results.
  virtual void ReturnError(const absl::Status& error) const = 0;
};

//===----------------------------------------------------------------------===//
// Result converter for functions without results (returning void).
//===----------------------------------------------------------------------===//

struct NoResultConverter : public ResultConverter {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  LogicalResult ReturnValue(unsigned, const Type*, const Type*,
                            void*) const final {
    assert(false && "no result converter must never be called");
    return failure();
  }

  void ReturnError(const absl::Status&) const final {}
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_RESULTS_H_
