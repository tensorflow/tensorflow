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

#include "tensorflow/compiler/xla/runtime/types.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/primitive_util.h"

namespace xla {
namespace runtime {

using xla::primitive_util::LowercasePrimitiveTypeName;

//===----------------------------------------------------------------------===//
// Pretty printing for canonical types.
//===----------------------------------------------------------------------===//

using absl::StrCat;
using absl::StrJoin;

static std::string FormatSizes(absl::Span<const int64_t> arr) {
  return arr.empty() ? "" : StrCat(StrJoin(arr, "x"), "x");
}

std::string AsyncTokenType::ToString() const { return "!async.token"; }

std::string AsyncValueType::ToString() const {
  return StrCat("!async.value<", value_type().ToString(), ">");
}

std::string RankedTensorType::ToString() const {
  return StrCat("tensor<", FormatSizes(sizes()),
                LowercasePrimitiveTypeName(element_type()), ">");
}

std::string UnrankedTensorType::ToString() const {
  return StrCat("tensor<*x", LowercasePrimitiveTypeName(element_type()), ">");
}

std::string MemrefType::ToString() const {
  return StrCat("memref<", FormatSizes(sizes()),
                LowercasePrimitiveTypeName(element_type()), ">");
}

std::string UnrankedMemrefType::ToString() const {
  return StrCat("memref<*x", LowercasePrimitiveTypeName(element_type()), ">");
}

std::string ExecutionContextOperandType::ToString() const {
  return "!rt.execution_context";
}

//===----------------------------------------------------------------------===//
// ABI definition for canonical types.
//===----------------------------------------------------------------------===//

using ArgumentAbi = Type::ArgumentAbi;
using ResultAbi = Type::ResultAbi;

// Async token returned as a pointer to the runtime async token.
absl::StatusOr<ResultAbi> AsyncTokenType::AsResult() const {
  return ResultAbi{sizeof(void*)};
}

// Async value returned as a pointer to the runtime async token.
absl::StatusOr<ResultAbi> AsyncValueType::AsResult() const {
  return ResultAbi{sizeof(void*)};
}

// Memref passed as an unrolled strided memref type.
absl::StatusOr<ArgumentAbi> MemrefType::AsArgument() const {
  return ArgumentAbi{3 + 2 * rank()};
}

// TODO(ezhulenev): We should query the size of the `StridedMemrefType`
// directly, however it introduces dependency on the MLIR C runner utils.
//
// Memrefs are returned as StridedMemref<T, rank> type:
//   basePtr, data, offset, sizes[rank], strides[rank]
absl::StatusOr<ResultAbi> MemrefType::AsResult() const {
  return ResultAbi{
      sizeof(void*) * 2 +           // pointers
      sizeof(int64_t) +             // offset
      sizeof(int64_t) * 2 * rank()  // sizes and strides
  };
}

// Execution context passed as a single opaque pointer.
absl::StatusOr<ArgumentAbi> ExecutionContextOperandType::AsArgument() const {
  return ArgumentAbi{1};
}

}  // namespace runtime
}  // namespace xla
