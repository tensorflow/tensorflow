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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/const_op_size.h"

#include <climits>

#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace quant {
namespace {

// For types that have varying sizes or difficult to determine the size of, each
// element is arbitrarily considered to be 4 bytes.
constexpr int64_t kAssumedNumBytesPerElem = 4;

int64_t GetSizeOfIntOrFloatConst(TF::ConstOp const_op) {
  const Type dtype = const_op.getDtype();
  const ElementsAttr const_value = const_op.getValue();

  const auto bytes_per_elem =
      static_cast<int64_t>(dtype.getIntOrFloatBitWidth() / CHAR_BIT);

  return bytes_per_elem * const_value.getNumElements();
}

int64_t GetSizeOfStringConst(TF::ConstOp const_op) {
  const ElementsAttr const_value = const_op.getValue();

  // This cast is guaranteed to succeed. See `ConvertToTensorProto` from
  // tensorflow/core/ir/importexport/convert_tensor.cc.
  const auto str_attr = cast<DenseStringElementsAttr>(const_value);

  // Sum the sizes of each string.
  return absl::c_accumulate(
      str_attr.getRawStringData(), 0,
      [](int64_t acc, const StringRef str_value) -> int64_t {
        return acc + str_value.size();
      });
}

// Arbitrarily calculate the size of const of type whose size is unkown or
// varying. Each element of such a type is considered to have
// `kAssumedNumBytesPerElem` bytes.
int64_t GetSizeOfUnsupportedTypeConst(TF::ConstOp const_op) {
  return kAssumedNumBytesPerElem * const_op.getValue().getNumElements();
}

}  // namespace

int64_t GetSizeInBytes(TF::ConstOp const_op) {
  const Type dtype = const_op.getDtype();

  if (dtype.isIntOrFloat()) {
    return GetSizeOfIntOrFloatConst(const_op);
  } else if (isa<TF::StringType>(dtype)) {
    return GetSizeOfStringConst(const_op);
  } else {
    return GetSizeOfUnsupportedTypeConst(const_op);
  }
}

}  // namespace quant
}  // namespace mlir
