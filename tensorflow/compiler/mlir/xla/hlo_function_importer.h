/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_HLO_FUNCTION_IMPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_HLO_FUNCTION_IMPORTER_H_

#include <unordered_map>

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloModule;
class HloComputation;
class HloInstruction;
class Shape;

// Helper class for importing HloComputations.
class HloFunctionImporter {
 public:
  static StatusOr<mlir::FuncOp> ImportFunction(
      mlir::ModuleOp module, mlir::Builder* builder,
      std::unordered_map<xla::HloComputation*, mlir::FuncOp>* function_map,
      xla::HloComputation* computation);

 private:
  HloFunctionImporter(
      mlir::ModuleOp module, mlir::Builder* builder,
      std::unordered_map<xla::HloComputation*, mlir::FuncOp>* function_map)
      : context_(module.getContext()),
        module_(module),
        builder_(builder),
        function_map_(function_map) {}

  StatusOr<mlir::FuncOp> ImportFunction(xla::HloComputation* computation);

  // Imports the given computation in the specified region.
  tensorflow::Status ImportComputation(HloComputation* computation,
                                       mlir::Region* region);

  // Imports instructions from the given computation in the specified block.
  // Assumes that the block already has correct arguments populated.
  tensorflow::Status ImportInstructions(HloComputation* computation,
                                        mlir::Block* block);

  // Imports an instruction.
  StatusOr<mlir::Operation*> ImportInstruction(xla::HloInstruction* instruction,
                                               mlir::OpBuilder* func_builder);

  // Gets the MLIR operand values from an HLO Instruction.
  StatusOr<llvm::SmallVector<mlir::Value*, 4>> GetOperands(
      xla::HloInstruction* instruction);

  // Converts xla Tensor type to the corresponding MLIR type.
  StatusOr<mlir::RankedTensorType> ConvertTensorType(const xla::Shape& shape);

  // Converts xla Primitive types to the corresponding MLIR type.
  StatusOr<mlir::Type> ConvertType(const xla::Shape& shape);

  // Returns the output type of an HloInstruction.
  StatusOr<mlir::Type> GetReturnType(xla::HloInstruction* instruction);

  // Takes a list of HloInstructions and generates the list of types used for
  // input, bypassing tuples to subsets.
  Status GetMlirTypes(const std::vector<xla::HloInstruction*>& instructions,
                      llvm::SmallVectorImpl<mlir::Type>* types);

  // Returns the Mlir Value for the corresponding HloInstruction.
  StatusOr<mlir::Value*> GetMlirValue(xla::HloInstruction* instruction);

  // Converts an XLA PrecisionConfig to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertPrecisionConfig(xla::HloInstruction* instruction);

  // Converts an XLA ComparisonDirection to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonDirection(
      xla::HloInstruction* instruction);

  // Converts the dimensions of an HLO instruction into an MLIR attribute.
  mlir::DenseIntElementsAttr ConvertDimensions(
      llvm::ArrayRef<tensorflow::int64> op_dimensions);

  // Converts Array ref to an DenseIntElementsAttr.
  mlir::DenseIntElementsAttr Convert(llvm::ArrayRef<int64_t> op_dimensions);

  // Converts Array ref to padding attribute. Input is a flattened list of
  // padding low and padding high for each of the spatial dimensions.
  mlir::NamedAttribute ConvertPadding(llvm::ArrayRef<int64_t> padding);

  // Converts the dot dimensions to attribute.
  mlir::NamedAttribute ConvertDotDimensionNumbers(
      const DotDimensionNumbers& dnums);

  // Converts the conv dimensions to attributes.
  mlir::NamedAttribute ConvertConvDimensionNumbers(
      const xla::ConvolutionDimensionNumbers& dnums);

  mlir::MLIRContext* context_;
  mlir::ModuleOp module_;
  mlir::Builder* builder_;

  // Mapping from HloComputation to the created MLIR function.
  std::unordered_map<xla::HloComputation*, mlir::FuncOp>* function_map_;

  // Mapping from HloInstructions to the associative MLIR values.
  std::unordered_map<xla::HloInstruction*, mlir::Value*> instruction_value_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_HLO_FUNCTION_IMPORTER_H_
