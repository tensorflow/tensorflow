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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_HLO_FUNCTION_IMPORTER_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_HLO_FUNCTION_IMPORTER_H_

#include <string>
#include <unordered_map>

#include "absl/types/optional.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo.pb.h"
#include "xla/status.h"
#include "xla/statusor.h"
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
  static StatusOr<mlir::func::FuncOp> ImportAsFunc(
      const xla::HloComputation& computation, mlir::SymbolTable& symbol_table,
      std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>*
          function_map,
      mlir::Builder* builder, bool is_main);

  // Imports the given hlo computation to the specified region. If
  // 'flatten_region_arg_tuple' is true, then flatten the tuple-typed region
  // argument(s) and return value(s).
  static Status ImportAsRegion(const xla::HloComputation& computation,
                               mlir::SymbolTable& symbol_table,
                               mlir::Region* region, mlir::Builder* builder,
                               bool flatten_region_arg_tuple = false);

  // Imports the given computation to the given place specified by `builder`.
  // `arguments` contains values for all parameters.
  static StatusOr<mlir::Value> ImportInstructions(
      const xla::HloComputation& computation,
      const llvm::SmallVectorImpl<mlir::Value>& arguments,
      mlir::SymbolTable& symbol_table, mlir::OpBuilder* builder);

  static StatusOr<mlir::Operation*> ImportInstruction(
      const xla::HloInstruction* instr,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::SymbolTable& symbol_table, mlir::OpBuilder* builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  static void SetLayoutForMlir(mlir::Operation* op, const Shape& shape,
                               llvm::StringRef attr_name);

  // TODO(b/179166199): move this to attribute_importer.h.
  // Converts XLA instruction source target pairs to MLIR attribute.
  static mlir::NamedAttribute ConvertSourceTargetPairs(
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      mlir::Builder* builder);

  // TODO(b/179166199): move this to attribute_importer.h.
  // Converts replica groups to attribute
  static mlir::NamedAttribute ConvertReplicaGroups(
      absl::Span<const ReplicaGroup> replica_groups, mlir::Builder* builder);

  // For mlir::IfOp or mlir::CaseOp, replace the uses of their region's block
  // arguments with 'implicit_operands'. Here | implicit_operands | == sum of
  // the number of arguments in all the regions in IfOp or CaseOp.
  void ReplaceBlockArgumentsWithImplicitOperands(
      mlir::Operation* op, llvm::ArrayRef<mlir::Value> implicit_operands);

  // Create a TupleOp using the results of 'op' if 'type' is a mlir::TupleType.
  // Otherwise, return 'op'.
  mlir::Operation* CreateTupleFromOpResults(mlir::OpBuilder* func_builder,
                                            mlir::Location loc,
                                            mlir::Operation* op,
                                            mlir::Type type);

  // FlattenTupleType flattens the types in (nested) tuple-type 'type' and
  // stores them in 'types'.
  static void FlattenTupleType(
      mlir::Type type, llvm::SmallVectorImpl<mlir::Type>& flattened_types);

  // FlattenTupleValue flattens the values in (nested) tuple-typed 'value' and
  // stores them in 'flattened_values'.
  static void FlattenTupleValue(
      mlir::OpBuilder* func_builder, mlir::Location loc, mlir::Value value,
      llvm::SmallVectorImpl<mlir::Value>& flattened_values);

  // CreateTupleValue creates a root TupleOp of (nested) tuple-type 'type' using
  // the non-tuple-typed values in 'flatten_values'.
  //
  // e.g., Given 'flatten_values': [V1, V2, V3] &'type': tuple<T1,tuple<T1,T2>>,
  //      The function returns %t2 such that:
  //       %t1 = mhlo.tuple(V2,V3) : (T2,T3) -> tuple<T2,T3>
  //       %t2 = mhlo.tuple(V1,%t1): (T1,tuple<T2,T3>) -> tuple<T1,tuple<T1,T2>>
  //
  // Note: 1. FlattenTupleValue and CreateTupleValue is a pair of functions to
  //          resp. flatten and create tuples in the exact same order.
  //       2. `flatten_values`, initially storing the flattened values, will be
  //          mutated to a 0-length array by the end of function invocation.
  static mlir::Value CreateTupleValue(
      mlir::OpBuilder* func_builder, mlir::Location loc,
      llvm::MutableArrayRef<mlir::Value>& flatten_values, mlir::Type type);

 private:
  HloFunctionImporter(mlir::SymbolTable& symbol_table,
                      std::unordered_map<const xla::HloComputation*,
                                         mlir::func::FuncOp>* function_map,
                      mlir::Builder* builder)
      : context_(symbol_table.getOp()->getContext()),
        symbol_table_(symbol_table),
        builder_(builder),
        function_map_(function_map) {
    context_->loadDialect<mlir::arith::ArithDialect>();
    context_->loadDialect<mlir::func::FuncDialect>();
    context_->loadDialect<mlir::mhlo::MhloDialect>();
    context_->loadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  }

  // Imports the given computation as a new function, if it hasn't been already
  // imported.
  StatusOr<mlir::func::FuncOp> ImportAsFunc(
      const xla::HloComputation& computation, bool is_main);

  // Imports the given computation in the specified region.
  Status ImportAsRegion(const HloComputation& computation, mlir::Region* region,
                        bool flatten_region_arg_tuple = false);

  // Imports instructions from the given computation in the specified block.
  // Assumes that the block already has correct arguments populated.
  Status ImportInstructions(const HloComputation& computation,
                            mlir::Block* block, bool flatten_region_arg_tuple);
  StatusOr<mlir::Value> ImportInstructionsImpl(
      const xla::HloComputation& computation,
      const llvm::SmallVectorImpl<mlir::Value>& arguments,
      mlir::OpBuilder* builder);

  // Imports an instruction.
  StatusOr<mlir::Operation*> ImportInstructionWithLayout(
      const xla::HloInstruction* instruction,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* func_builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  StatusOr<mlir::Operation*> ImportInstructionImpl(
      const HloInstruction* instruction,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* func_builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  // Gets the MLIR operand values from an HLO Instruction.
  StatusOr<llvm::SmallVector<mlir::Value, 4>> GetOperands(
      const xla::HloInstruction* instruction);

  // Converts xla Tensor type to the corresponding MLIR type.
  StatusOr<mlir::RankedTensorType> ConvertTensorType(const xla::Shape& shape);

  // Converts an XLA shape/layout to the corresponding MLIR layout, in
  // flattened_attr, while flattening the tuple layout.
  Status ConvertShapeToMlirLayout(
      const xla::Shape& shape,
      llvm::SmallVectorImpl<mlir::Attribute>& flattened_attr);

  // Returns the output type of an HloInstruction.
  StatusOr<mlir::Type> GetReturnType(const xla::HloInstruction* instruction);

  // Takes a list of HloInstructions and generates the list of types used for
  // input, bypassing tuples to subsets.
  Status GetMlirTypes(absl::Span<const HloInstruction* const> instructions,
                      llvm::SmallVectorImpl<mlir::Type>* types);

  // Returns the Mlir Value for the corresponding HloInstruction.
  StatusOr<mlir::Value> GetMlirValue(const xla::HloInstruction* instruction);

  // Converts an XLA ComparisonDirection to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonDirection(
      ComparisonDirection direction);

  // Converts an XLA Comparison::Type to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonType(Comparison::Type type);

  // Converts an XLA CustomCallSchedule to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertCustomCallSchedule(
      xla::CustomCallSchedule schedule);

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

  // Converts channel id to attribute
  mlir::NamedAttribute ConvertChannelHandle(std::optional<int64_t> channel_id);

  // Convert use global device ids flag to attribute
  mlir::NamedAttribute ConvertUseGlobalDeviceIds();

  // Converts channel handle to attribute
  mlir::NamedAttribute ConvertChannelHandle(const xla::ChannelHandle& channel);

  // ============
  // Imports an old-style async start op. E.g. an HLO all-gather-start
  // instruction is imported as an async-start associated with an all-gather
  // computation.
  //
  // Eventually, old-style async ops (e.g. all-gather-start) and new-style async
  // ops (i.e. async-start, async-update and async-done) will converge on the
  // HLO side, so we decided to not introduce new MHLO ops for all-gather-start
  // and friends.
  //
  // In the end, there may be new ops added in the old-style because they're not
  // compatible with the new-style async semantics, but those should be handled
  // on their own, rather than this function which "upgrades" ops to the
  // new-style async API.
  // ============
  template <typename SyncOp>
  StatusOr<mlir::Operation*> ImportOldStyleAsyncStart(
      llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
      const llvm::SmallVectorImpl<mlir::Value>& operands, mlir::Location loc,
      mlir::Type result_type, mlir::OpBuilder* func_builder,
      std::string func_name, std::function<Status(SyncOp)> mutate_op);

  // Imports an old-style async done op
  StatusOr<mlir::Operation*> ImportOldStyleAsyncDone(
      llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
      const llvm::SmallVectorImpl<mlir::Value>& operands, mlir::Location loc,
      mlir::Type result_type, mlir::OpBuilder* func_builder);

  mlir::MLIRContext* context_;

  // SymbolTable to which new functions should be inserted.
  mlir::SymbolTable& symbol_table_;

  mlir::Builder* builder_;

  // Mapping from HloComputation to the created MLIR function.
  std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>*
      function_map_;

  // Mapping from HloInstructions to the associative MLIR values.
  std::unordered_map<const xla::HloInstruction*, mlir::Value>
      instruction_value_map_;
};

// Returns a StringAttr that carries a prettyprinted representation of the
// given HLO C++ sharding.
// Always succeeds and returns a non-empty attribute.
mlir::Attribute ConvertSharding(const xla::HloSharding& sharding,
                                mlir::Builder* builder);

// Returns a StringAttr that carries a prettyprinted representation of the
// given HLO proto sharding.
// Will fail and return an empty attribute if the proto sharding cannot be
// converted to the C++ sharding.
mlir::Attribute ConvertSharding(const xla::OpSharding& sharding,
                                mlir::Builder* builder);

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_HLO_FUNCTION_IMPORTER_H_
