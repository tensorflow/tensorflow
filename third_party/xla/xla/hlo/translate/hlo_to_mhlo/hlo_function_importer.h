/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_HLO_FUNCTION_IMPORTER_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_HLO_FUNCTION_IMPORTER_H_

#include <cstdint>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

class HloModule;
class HloComputation;
class HloInstruction;
class Shape;

// HLO bounded dynamic shapes can be converted to either MLIR dynamic shapes
// (which lose the bound information) or casted to static shape using the
// bounds.
enum class DynamicShapeHandlingMode { kDynamic, kConvertToStatic };

// Helper class for importing HloComputations.
class HloFunctionImporter {
 public:
  // Imports the given computation as a function in the given symbol table and
  // returns the FuncOp. This also imports any computations referred by
  // instructions in this computation.
  static absl::StatusOr<mlir::func::FuncOp> ImportAsFunc(
      const HloComputation& computation, mlir::SymbolTable& symbol_table,
      std::unordered_map<const HloComputation*, mlir::func::FuncOp>*
          function_map,
      mlir::Builder* builder, bool is_main,
      bool flatten_computation_args_result = false);

  // Imports the given hlo computation to the specified region.
  //
  // Flattens the tuple-typed region argument(s) and return value(s).
  static absl::Status ImportAsRegion(
      const HloComputation& computation, mlir::SymbolTable& symbol_table,
      mlir::Region* region, mlir::Builder* builder,
      bool flatten_computation_args_result = false);

  // Imports the given computation to the given place specified by `builder`.
  // `arguments` contains values for all parameters.
  static absl::StatusOr<mlir::Value> ImportInstructions(
      const HloComputation& computation,
      const llvm::SmallVectorImpl<mlir::Value>& arguments,
      mlir::SymbolTable& symbol_table, mlir::OpBuilder* builder,
      bool flatten_computation_args_result = false);

  static absl::StatusOr<mlir::Operation*> ImportInstruction(
      const HloInstruction* instr,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::SymbolTable& symbol_table, mlir::OpBuilder* builder,
      bool flatten_computation_args_result = false,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  static void SetLayoutForMlir(mlir::Operation* op, const Shape& shape,
                               llvm::StringRef attr_name);

  // For mlir::IfOp or mlir::CaseOp, replace the uses of their region's block
  // arguments with 'implicit_operands'. Here | implicit_operands | == sum of
  // the number of arguments in all the regions in IfOp or CaseOp.
  void ReplaceBlockArgumentsWithImplicitOperands(
      mlir::Operation* op, llvm::ArrayRef<mlir::Value> implicit_operands);

  // FlattenTupleType flattens the types in (nested) tuple-type 'type' and
  // stores them in 'flattened_types'.
  static void FlattenTupleType(
      mlir::Type type, llvm::SmallVectorImpl<mlir::Type>& flattened_types);

  // FlattenTupleValue flattens the values in (nested) tuple-typed 'value' and
  // stores them in 'flattened_values'.
  static void FlattenTupleValue(
      mlir::OpBuilder* func_builder, mlir::Location loc, mlir::Value value,
      llvm::SmallVectorImpl<mlir::Value>& flattened_values);

  // FlattenTupleValues flattens the values in (nested) tuple-typed 'values' and
  // returns the flattened values.
  static llvm::SmallVector<mlir::Value> FlattenTupleValues(
      mlir::OpBuilder* func_builder, mlir::Location loc,
      mlir::ValueRange values, std::optional<int> reserve_size = std::nullopt);

 private:
  HloFunctionImporter(mlir::SymbolTable& symbol_table,
                      std::unordered_map<const HloComputation*,
                                         mlir::func::FuncOp>* function_map,
                      mlir::Builder* builder,
                      bool flatten_computation_args_result)
      : context_(symbol_table.getOp()->getContext()),
        symbol_table_(symbol_table),
        builder_(builder),
        function_map_(function_map),
        flatten_computation_args_result_(flatten_computation_args_result) {
    context_->loadDialect<mlir::arith::ArithDialect>();
    context_->loadDialect<mlir::func::FuncDialect>();
    context_->loadDialect<mlir::mhlo::MhloDialect>();
    context_->loadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  }

  // Imports the given computation as a new function, if it hasn't been already
  // imported.
  absl::StatusOr<mlir::func::FuncOp> ImportAsFunc(
      const HloComputation& computation, bool is_main);

  // Imports the given computation in the specified region.
  absl::Status ImportAsRegion(const HloComputation& computation,
                              mlir::Region* region);

  // Imports instructions from the given computation in the specified block.
  // Assumes that the block already has correct arguments populated.
  absl::Status ImportInstructions(const HloComputation& computation,
                                  mlir::Block* block);
  absl::StatusOr<mlir::Value> ImportInstructionsImpl(
      const HloComputation& computation,
      const llvm::SmallVectorImpl<mlir::Value>& arguments,
      mlir::OpBuilder* builder);

  // Imports an instruction.
  absl::StatusOr<mlir::Operation*> ImportInstructionWithLayout(
      const HloInstruction* instruction,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* func_builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  absl::StatusOr<mlir::Operation*> ImportInstructionImpl(
      const HloInstruction* instruction,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* func_builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  // Gets the MLIR operand values from an HLO Instruction.
  absl::StatusOr<llvm::SmallVector<mlir::Value, 4>> GetOperands(
      const HloInstruction* instruction);

  // Converts xla Tensor type to the corresponding MLIR type.
  absl::StatusOr<mlir::RankedTensorType> ConvertTensorType(const Shape& shape);

  // Converts an XLA shape/layout to the corresponding MLIR layout, in
  // flattened_attr, while flattening the tuple layout.
  absl::Status ConvertShapeToMlirLayout(
      const Shape& shape,
      llvm::SmallVectorImpl<mlir::Attribute>& flattened_attr);

  // Returns the output type of an HloInstruction.
  absl::StatusOr<mlir::Type> GetReturnType(const HloInstruction* instruction);

  // Takes a list of HloInstructions and generates the list of types used for
  // input, bypassing tuples to subsets.
  absl::Status GetMlirTypes(
      absl::Span<const HloInstruction* const> instructions,
      llvm::SmallVectorImpl<mlir::Type>* types);

  // Returns the Mlir Value for the corresponding HloInstruction.
  absl::StatusOr<mlir::Value> GetMlirValue(const HloInstruction* instruction);

  // TODO(b/179166199): Move attribute converters to attribute_importer.
  // Converts an XLA ComparisonDirection to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonDirection(
      ComparisonDirection direction);

  // Converts an XLA Comparison::Type to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonType(Comparison::Type type);

  // Converts an XLA CustomCallSchedule to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertCustomCallSchedule(CustomCallSchedule schedule);

  // Converts the dimensions of an HLO instruction into an MLIR attribute.
  mlir::DenseIntElementsAttr ConvertDimensions(
      absl::Span<const int64_t> op_dimensions);

  // Converts Array ref to an DenseIntElementsAttr.
  mlir::DenseIntElementsAttr Convert(llvm::ArrayRef<int64_t> elements);

  // Converts Array ref of bools to a DenseIntElementsAttr of I1 type.
  mlir::DenseIntElementsAttr Convert(llvm::ArrayRef<bool> elements);

  // Converts Array ref to padding attribute. Input is a flattened list of
  // padding low and padding high for each of the spatial dimensions.
  mlir::NamedAttribute ConvertPadding(llvm::ArrayRef<int64_t> padding);

  mlir::MLIRContext* context_;

  // SymbolTable to which new functions should be inserted.
  mlir::SymbolTable& symbol_table_;

  mlir::Builder* builder_;

  // Mapping from HloComputation to the created MLIR function.
  std::unordered_map<const HloComputation*, mlir::func::FuncOp>* function_map_;

  // Mapping from HloInstructions to the associative MLIR values.
  std::unordered_map<const HloInstruction*, mlir::Value> instruction_value_map_;

  bool flatten_computation_args_result_;
};

// Returns a StringAttr that carries a prettyprinted representation of the
// given HLO C++ input_output_alias_config.
// Always succeeds and returns a non-empty attribute.
mlir::Attribute ConvertInputOutputAlias(const HloInputOutputAliasConfig& alias,
                                        mlir::Builder* builder);

// Returns a StringAttr that carries a prettyprinted representation of the
// given HLO C++ sharding.
// Always succeeds and returns a non-empty attribute.
mlir::Attribute ConvertSharding(const HloSharding& sharding,
                                mlir::Builder* builder);

// Returns a StringAttr that carries a prettyprinted representation of the
// given HLO proto sharding.
// Will fail and return an empty attribute if the proto sharding cannot be
// converted to the C++ sharding.
mlir::Attribute ConvertSharding(const OpSharding& sharding,
                                mlir::Builder* builder);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_HLO_FUNCTION_IMPORTER_H_
