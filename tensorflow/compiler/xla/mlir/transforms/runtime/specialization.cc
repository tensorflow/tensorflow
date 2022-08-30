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

#include "tensorflow/compiler/xla/mlir/transforms/runtime/specialization.h"

#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/runtime/type_converter.h"
#include "tensorflow/compiler/xla/mlir/utils/runtime/constraints.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/symbolic_shape.h"

namespace xla {
namespace runtime {

using llvm::ArrayRef;
using llvm::dyn_cast;

using absl::InvalidArgumentError;
using absl::Status;
using absl::StatusOr;
using absl::StrCat;

using SymbolicShape = SymbolicShapesResolver::SymbolicShape;

static Status VerifyMemrefOperand(unsigned index, mlir::ShapedType shaped,
                                  const MemrefDesc& memref) {
  auto element_ty = TypeConverter::ConvertElementType(shaped.getElementType());
  if (!element_ty.ok()) element_ty.status();

  // TODO(ezhulenev): Pass an instance of TypeConverter so we can convert shaped
  // type to the corresponding run-time type. For now we convert all shaped
  // types to memrefs, because for the verification function it doesn't really
  // matter if it's a tensor or a memref.

  // We do not support unranked memrefs at runtime, however we need to verify
  // operand types when we do compiled kernel specialization to shape.
  if (shaped.hasRank()) {
    MemrefType type(shaped.getShape(), *element_ty);
    if (auto st = VerifyMemrefArgument(index, type, memref); !st.ok())
      return st;
  } else {
    UnrankedMemrefType type(*element_ty);
    if (auto st = VerifyMemrefArgument(index, type, memref); !st.ok())
      return st;
  }

  return absl::OkStatus();
}

// Return input `type` specialized to the argument and its symbolic shape.
static StatusOr<mlir::Type> SpecializeOperandType(
    unsigned index, mlir::Type type, const Argument& argument,
    const SymbolicShape& symbolic_shape) {
  // We do not yet support specializing non-memref arguments.
  auto* memref_arg = dyn_cast<MemrefDesc>(&argument);
  if (!memref_arg) {
    if (!symbolic_shape.empty())
      return InvalidArgumentError(StrCat(
          "unexpected symbolic shape for argument: ", argument.ToString()));
    return type;
  }

  // Replace all symbolic dimensions with dynamic dimension.
  auto shape = SymbolicShapesResolver::Normalize(symbolic_shape);

  if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
    if (auto st = VerifyMemrefOperand(index, memref, *memref_arg); !st.ok())
      return st;
    return mlir::MemRefType::get(shape, memref.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::RankedTensorType>()) {
    if (auto st = VerifyMemrefOperand(index, tensor, *memref_arg); !st.ok())
      return st;
    return mlir::RankedTensorType::get(shape, tensor.getElementType());
  }

  if (auto tensor = type.dyn_cast<mlir::UnrankedTensorType>()) {
    if (auto st = VerifyMemrefOperand(index, tensor, *memref_arg); !st.ok())
      return st;
    return mlir::RankedTensorType::get(shape, tensor.getElementType());
  }

  return InvalidArgumentError(
      StrCat("Unsupported input type: ", debugString(type)));
}

// Gets (copies) the values from `desc`, returning them in a DenseElementsAttr.
// If it cannot extract the values, returns an empty attribute.
static mlir::DenseElementsAttr GetMemrefValues(mlir::Builder& builder,
                                               mlir::TensorType operand_type,
                                               const MemrefDesc& desc) {
  size_t rank = desc.rank();
  if (rank != 0 && rank != 1) return {};

  llvm::SmallVector<mlir::Attribute> attributes;
  size_t num_values = rank == 0 ? 1 : desc.size(0);
  switch (desc.dtype()) {
    case PrimitiveType::S32: {
      const auto* data = static_cast<int32_t*>(desc.data());
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder.getI32IntegerAttr(data[i]));
      }
    } break;
    case PrimitiveType::S64: {
      const auto* data = static_cast<int64_t*>(desc.data());
      for (int i = 0; i < num_values; ++i) {
        attributes.push_back(builder.getI64IntegerAttr(data[i]));
      }
    } break;
    default:
      return {};
  }

  // Update operand type to a ranked tensor type with statically known shape.
  auto element_type = operand_type.getElementType();
  auto ranked_tensor = mlir::RankedTensorType::get(
      {desc.sizes().begin(), desc.sizes().size()}, element_type);

  return mlir::DenseElementsAttr::get(ranked_tensor, attributes);
}

Status SpecializeFunction(mlir::func::FuncOp func, ArgumentsRef arguments,
                          ArrayRef<SymbolicShape> symbolic_shapes,
                          ArrayRef<ArgumentConstraint> constraints,
                          const SpecializationListener* listener) {
  mlir::MLIRContext* ctx = func.getContext();

  unsigned num_inputs = func.getNumArguments();

  // Specialize all function inputs to the given arguments.
  llvm::SmallVector<mlir::Type> specialized_inputs(num_inputs);
  for (unsigned i = 0; i < num_inputs; ++i) {
    auto specialized =
        SpecializeOperandType(i, func.getFunctionType().getInput(i),
                              arguments[i], symbolic_shapes[i]);
    if (!specialized.ok()) return specialized.status();
    specialized_inputs[i] = *specialized;
  }

  // Update function type to a new specialized one.
  auto specialized = mlir::FunctionType::get(
      ctx, specialized_inputs, func.getFunctionType().getResults());
  func.setType(specialized);

  // Update function entry block arguments.
  mlir::Block& entry_block = func.getBlocks().front();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(&entry_block);
  mlir::Location loc = func.getLoc();

  // Forward original block arguments to arguments with specialized type. We
  // need to insert casts to ensure the users still get the correct type and
  // avoid illegal IR. This can be optimized away by the user-provided
  // specialization pipeline, e.g., in Tensorflow these casts will be optimized
  // away by the shape inference pass.
  for (int i = 0; i < num_inputs; ++i) {
    mlir::Value new_arg = entry_block.addArgument(specialized_inputs[i], loc);
    mlir::Value old_arg = entry_block.getArgument(i);
    if (new_arg.getType() != old_arg.getType()) {
      new_arg =
          builder.create<mlir::tensor::CastOp>(loc, old_arg.getType(), new_arg);
    }
    old_arg.replaceAllUsesWith(new_arg);
  }

  // Erase all the original block arguments.
  llvm::BitVector erase_block_args(entry_block.getNumArguments());
  erase_block_args.set(0, num_inputs);
  entry_block.eraseArguments(erase_block_args);

  // Add symbolic shapes as arguments attributes.
  for (unsigned i = 0; i < num_inputs; ++i) {
    const SymbolicShape& shape = symbolic_shapes[i];
    int64_t rank = shape.size();

    // Skip statically known shapes.
    if (llvm::all_of(shape, [](int64_t dim) { return dim >= 0; })) continue;

    // Symbolic shape attribute stored as 1d tensor attribute.
    auto i64 = mlir::IntegerType::get(ctx, 64);
    auto tensor = mlir::RankedTensorType::get({rank}, i64);

    // Create i64 attributes from the symbolic shape values.
    llvm::SmallVector<mlir::Attribute> values(rank);
    for (unsigned d = 0; d < rank; ++d)
      values[d] = mlir::IntegerAttr::get(i64, shape[d]);

    func.setArgAttr(i, kSymbolicShapeAttrName,
                    mlir::DenseElementsAttr::get(tensor, values));
  }

  // Sink small constants into the function body.
  builder.setInsertionPointToStart(&func.getBody().front());
  for (int i = 0; i < constraints.size(); ++i) {
    if (constraints[i] != ArgumentConstraint::kValue) continue;

    // We only support sinking of Tensor arguments into the function body.
    mlir::Type input = func.getFunctionType().getInput(i);
    mlir::TensorType tensor = input.dyn_cast<mlir::TensorType>();
    if (!tensor || !SupportsValueSpecialization(tensor)) {
      return InvalidArgumentError(StrCat(
          "non-sinkable operand was marked for sinking: ", debugString(input)));
    }

    // Value specialized tensors must be passed as memref arguments.
    auto* memref = dyn_cast<MemrefDesc>(&arguments[i]);
    if (!memref) {
      return InvalidArgumentError(
          StrCat("non-sinkable argument was marked for sinking: ",
                 arguments[i].ToString()));
    }

    // Get the argument value from the runtime memref argument.
    mlir::DenseElementsAttr value = GetMemrefValues(builder, tensor, *memref);
    if (!value) {
      return InvalidArgumentError(
          StrCat("cannot get value from argument type: ", debugString(input)));
    }

    auto cst =
        builder.create<mlir::arith::ConstantOp>(loc, value.getType(), value);
    entry_block.getArgument(i).replaceAllUsesWith(cst);

    if (listener) listener->notifyValueSpecialized(i, value.getType(), value);
  }

  if (listener) {
    llvm::SmallVector<mlir::DictionaryAttr> specialized_attrs;
    func.getAllArgAttrs(specialized_attrs);
    listener->notifyModuleSpecialized(specialized_inputs, specialized_attrs);
  }

  return absl::OkStatus();
}

}  // namespace runtime
}  // namespace xla
