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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_common {

template <typename OpTy>
mlir::LogicalResult VerifyExecuteOpCommon(OpTy op) {
  auto op_attr_array = op.op_attrs().getValue();
  for (auto op_attr : op_attr_array) {
    auto key_value = op_attr.template dyn_cast<mlir::ArrayAttr>();
    if (!key_value || key_value.getValue().size() != 2 ||
        !key_value.getValue()[0].template isa<mlir::StringAttr>())
      return op.emitOpError() << "each op_attr should be a key-value pair, "
                                 "where the key is a string";
  }
  return mlir::success();
}

template <typename OpTy>
mlir::LogicalResult VerifyFallbackExecuteOp(OpTy op) {
  auto result = VerifyExecuteOpCommon(op);
  if (failed(result)) return result;

  // Verify function attributes.
  auto op_func_attr_array = op.op_func_attrs().getValue();
  for (auto op_attr : op_func_attr_array) {
    auto key_value = op_attr.template dyn_cast<mlir::ArrayAttr>();
    if (!key_value || key_value.getValue().size() != 2 ||
        !key_value.getValue()[0].template isa<mlir::StringAttr>() ||
        !key_value.getValue()[1].template isa<mlir::StringAttr>())
      return op.emitOpError() << "each op_func_attr should be a key-value "
                                 "pair, where both the key and the value are "
                                 "strings";
  }
  return mlir::success();
}

template <typename OpTy>
void PrintExecuteOpFuncAttribute(mlir::OpAsmPrinter &p, OpTy op) {
  auto op_func_attrs = op.op_func_attrs();
  if (!op_func_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = attr.cast<mlir::ArrayAttr>().getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << key.cast<mlir::StringAttr>().getValue();
      p << " = ";
      p << value;
    };

    auto op_func_attr_array = op_func_attrs.getValue();
    p << " {";
    llvm::interleaveComma(op_func_attr_array, p, print_key_value);
    p << '}';
  }
}

template <typename OpTy>
void PrintExecuteOpCommon(mlir::OpAsmPrinter &p, OpTy op) {
  auto op_attrs = op.op_attrs();
  if (!op_attrs.empty()) {
    auto print_key_value = [&](mlir::Attribute attr) {
      auto key_value = attr.cast<mlir::ArrayAttr>().getValue();
      auto key = key_value[0];
      auto value = key_value[1];

      p << key.cast<mlir::StringAttr>().getValue();
      p << " = ";
      p << value;
    };

    auto op_attr_array = op_attrs.getValue();
    p << " {";
    llvm::interleaveComma(op_attr_array, p, print_key_value);
    p << '}';
  }
}

void GetExecuteOpAttrsCommon(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::Attribute> op_attr_array,
    llvm::SmallVectorImpl<std::pair<llvm::StringRef, mlir::Attribute>>
        *op_attrs);

struct ParseExecuteOpOptions {
  bool has_chain = false;
  bool has_key = false;
  bool has_device = false;
  bool has_func_attr = false;
  bool has_cost = false;
};

mlir::ParseResult ParseExecuteOpCommon(mlir::OpAsmParser &parser,
                                       mlir::Builder &builder,
                                       mlir::OperationState &result,
                                       mlir::Type tensor_type,
                                       const ParseExecuteOpOptions &options);
}  // namespace fallback_common
}  // namespace tfrt

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_COMMON_H_
