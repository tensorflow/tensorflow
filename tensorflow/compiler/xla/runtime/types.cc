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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace xla {
namespace runtime {

//===----------------------------------------------------------------------===//
// Pretty printing for canonical types.
//===----------------------------------------------------------------------===//

using llvm::raw_ostream;

static raw_ostream& operator<<(raw_ostream& os, absl::Span<const int64_t> arr) {
  auto str = llvm::map_range(arr, [](int64_t i) { return std::to_string(i); });
  return os << llvm::join(str, "x") << (arr.empty() ? "" : "x");
}

raw_ostream& AsyncTokenType::print(raw_ostream& os) const {
  return os << "!async.token";
}

raw_ostream& AsyncValueType::print(raw_ostream& os) const {
  return os << "!async.value<" << value_type() << ">";
}

raw_ostream& RankedTensorType::print(raw_ostream& os) const {
  return os << "tensor<" << sizes() << element_type() << ">";
}

raw_ostream& UnrankedTensorType::print(raw_ostream& os) const {
  return os << "tensor<*x" << element_type() << ">";
}

raw_ostream& MemrefType::print(raw_ostream& os) const {
  return os << "memref<" << sizes() << element_type() << ">";
}

raw_ostream& UnrankedMemrefType::print(raw_ostream& os) const {
  return os << "memref<*x" << element_type() << ">";
}

raw_ostream& KernelContextOperandType::print(raw_ostream& os) const {
  return os << "!rt.kernel_context";
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

// Kernel context passed as a single opaque pointer.
absl::StatusOr<ArgumentAbi> KernelContextOperandType::AsArgument() const {
  return ArgumentAbi{1};
}

}  // namespace runtime
}  // namespace xla
