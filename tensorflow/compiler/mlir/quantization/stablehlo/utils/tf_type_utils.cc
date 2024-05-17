/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/tf_type_utils.h"

#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir::quant::tensorflow {

bool IsTFQintType(const Type type) {
  return mlir::isa<TF::Qint8Type, TF::Qint16Type, TF::Qint32Type,
                   TF::Quint8Type, TF::Quint16Type>(type);
}

Type GetIntTypeFromTFQint(const Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<TF::Qint8Type>(
          [&type](Type) { return IntegerType::get(type.getContext(), 8); })
      .Case<TF::Qint16Type>(
          [&type](Type) { return IntegerType::get(type.getContext(), 16); })
      .Case<TF::Qint32Type>(
          [&type](Type) { return IntegerType::get(type.getContext(), 32); })
      .Case<TF::Quint8Type>([&type](Type) {
        return IntegerType::get(type.getContext(), 8,
                                IntegerType::SignednessSemantics::Unsigned);
      })
      .Case<TF::Quint16Type>([&type](Type) {
        return IntegerType::get(type.getContext(), 16,
                                IntegerType::SignednessSemantics::Unsigned);
      })
      .Default([&type](Type) { return type; });
}

FailureOr<mlir::DenseElementsAttr> GetDenseAttrFromTensorProtoAttr(
    const llvm::StringRef mangled_tensor_proto, TensorType tensor_type) {
  ::tensorflow::TensorProto tensor_proto;
  absl::Status status = ::tensorflow::mangling_util::DemangleTensor(
      mangled_tensor_proto, &tensor_proto);
  if (!status.ok()) {
    return failure();
  }

  ::tensorflow::Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return failure();
  }

  if (t.dtype() == ::tensorflow::DT_QINT8) {
    const auto arr = t.flat<::tensorflow::qint8>();
    return mlir::DenseElementsAttr::get(
        tensor_type.clone(IntegerType::get(tensor_type.getContext(), 8)),
        llvm::ArrayRef(arr.data(), arr.size()));
  } else if (t.dtype() == ::tensorflow::DT_QINT32) {
    const auto arr = t.flat<::tensorflow::qint32>();
    return mlir::DenseElementsAttr::get(
        tensor_type.clone(IntegerType::get(tensor_type.getContext(), 32)),
        llvm::ArrayRef(arr.data(), arr.size()));
  } else {
    return failure();
  }
}

bool IsTFUniformQuantizedOp(Operation *op) {
  return llvm::isa<
      // clang-format off
      // go/keep-sorted start
      TF::UniformDequantizeOp,
      TF::UniformQuantizeOp,
      TF::UniformQuantizedAddOp,
      TF::UniformQuantizedClipByValueOp,
      TF::UniformQuantizedConvolutionHybridOp,
      TF::UniformQuantizedConvolutionOp,
      TF::UniformQuantizedDotHybridOp,
      TF::UniformQuantizedDotOp,
      TF::UniformRequantizeOp
      // go/keep-sorted end
      // clang-format on
      >(op);
}

}  // namespace mlir::quant::tensorflow
