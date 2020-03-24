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
#ifndef TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

// Provides a way to construct xla_hlo dialect ops in MLIR using XlaBuilder
// interface.
//
// Requires that all XlaOp arguments are either returned by any of the builder
// method or constructed using MakeXlaOp method in this builder.
//
// TODO(hinsu): Support more ops and utility functions to set special attributes
// like OpMetadata and Sharding.
class MlirHloBuilder : public XlaBuilder {
 public:
  // Constructs builder for the given function. New operations are added to the
  // beginning of the function, if it is non empty and has a block.
  explicit MlirHloBuilder(mlir::FuncOp func)
      : XlaBuilder(func.getName().str()),
        builder_(&func.getBody()),
        loc_(builder_.getUnknownLoc()) {}

  // TODO(hinsu): Add a constructor to build a new MLIR function from scratch
  // and override Build methods.

  MlirHloBuilder(const MlirHloBuilder&) = delete;
  MlirHloBuilder& operator=(const MlirHloBuilder&) = delete;

  ~MlirHloBuilder() override;

  // Wraps the given MLIR value under an XlaOp instance. Note that all HLO
  // operations returns exactly one result therefore each op has an XlaOp
  // wrapping result of the op.
  //
  // Returns an error if the HLO dialect doesn't support type of the given
  // value.
  StatusOr<XlaOp> MakeXlaOp(mlir::Value val);

  // Returns value corresponding to the given op.
  //
  // Requires that the op was created by this builder.
  mlir::Value GetValue(XlaOp op) {
    void* ptr = reinterpret_cast<void*>(op.handle());
    return mlir::Value::getFromOpaquePointer(ptr);
  }

  // Sets location for newly built ops, until reset.
  void SetLocation(mlir::Location loc) { loc_ = loc; }

  // Update insertion point so that newly built ops are inserted before the
  // given op in order, until reset.
  void setInsertionPoint(mlir::Operation* op) {
    builder_.setInsertionPoint(op);
  }

  // Returns the shape of the given op.
  StatusOr<const Shape*> GetShapePtr(XlaOp op) const override;

 private:
  XlaOp ConstantLiteral(const LiteralSlice& literal) override;

  StatusOr<XlaOp> ReshapeInternal(const Shape& shape, XlaOp operand,
                                  int64 inferred_dimension) override;

  StatusOr<XlaOp> InDimBroadcast(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64> broadcast_dimensions) override;

  XlaOp BinaryOpNoBroadcast(
      HloOpcode binop, const Shape& shape, XlaOp lhs, XlaOp rhs,
      absl::optional<ComparisonDirection> direction) override;

  StatusOr<XlaOp> AddOpWithShape(HloOpcode opcode, const Shape& shape,
                                 absl::Span<const XlaOp> operands) override;

  // Creates HLO dialect op and returns the result as an XlaOp.
  StatusOr<XlaOp> CreateOp(const std::string& op_name, const Shape& shape,
                           llvm::ArrayRef<XlaOp> operands,
                           llvm::ArrayRef<mlir::NamedAttribute> attributes);

  mlir::OpBuilder builder_;
  mlir::Location loc_;

  absl::flat_hash_map<int64, std::unique_ptr<Shape>> handle_to_shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
