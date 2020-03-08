/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

static std::string GetMlirOpName(HloOpcode opcode) {
  std::string op_name = HloOpcodeString(opcode);
  absl::c_replace(op_name, '-', '_');
  return mlir::xla_hlo::XlaHloDialect::getDialectNamespace().str() + "." +
         op_name;
}

static std::string ToString(mlir::Type ty) {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  ty.print(sstream);
  sstream.flush();
  return str;
}

MlirHloBuilder::~MlirHloBuilder() = default;

StatusOr<XlaOp> MlirHloBuilder::MakeXlaOp(mlir::Value val) {
  mlir::Type ty = val.getType();
  auto shape = std::make_unique<Shape>(TypeToShape(ty));
  if (shape->element_type() == PrimitiveType::PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument("unsupported type: %s", ToString(ty).c_str());
  }

  int64 handle = reinterpret_cast<int64>(val.getAsOpaquePointer());
  handle_to_shape_[handle] = std::move(shape);
  return XlaOp(handle, this);
}

XlaOp MlirHloBuilder::UnaryOp(HloOpcode unop, XlaOp operand) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* operand_shape, GetShapePtr(operand));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferUnaryOpShape(unop, *operand_shape));

    mlir::Value value = GetValue(operand);
    mlir::OperationState state(loc_, GetMlirOpName(unop));
    state.addOperands(value);
    TF_ASSIGN_OR_RETURN(
        mlir::Type ty,
        ConvertShapeToType<mlir::RankedTensorType>(shape, builder_));
    state.addTypes(ty);
    mlir::Operation* op = builder_.createOperation(state);
    return MakeXlaOp(op->getResult(0));
  });
}

StatusOr<const Shape*> MlirHloBuilder::GetShapePtr(XlaOp op) const {
  TF_RETURN_IF_ERROR(first_error());
  TF_RETURN_IF_ERROR(CheckOpBuilder(op));
  auto it = handle_to_shape_.find(op.handle());
  if (it == handle_to_shape_.end()) {
    return InvalidArgument("No XlaOp with handle %d", op.handle());
  }
  return it->second.get();
}

}  // namespace xla
