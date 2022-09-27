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
#include <string>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Provides a way to construct mhlo dialect ops in MLIR using XlaBuilder
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
  explicit MlirHloBuilder(mlir::func::FuncOp func)
      : XlaBuilder(func.getName().str()),
        builder_(&func.getBody()),
        loc_(builder_.getUnknownLoc()),
        build_functions_(false) {}

  // TODO(hinsu): Add a constructor to build a new MLIR function from scratch
  // and override Build methods.

  MlirHloBuilder(std::string name, mlir::OpBuilder builder, mlir::Location loc,
                 bool build_functions)
      : XlaBuilder(name),
        builder_(builder),
        loc_(loc),
        build_functions_(build_functions) {}

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

  // Returns MLIR values corresponding to the given XLA ops.
  //
  // Requires that the ops were created by this builder.
  std::vector<mlir::Value> GetValues(absl::Span<const XlaOp> ops) {
    std::vector<mlir::Value> values;
    for (auto xla_op : ops) {
      values.push_back(GetValue(xla_op));
    }
    return values;
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

  // Creates the given op at the current location.
  template <typename OpTy, typename... Args>
  OpTy create(Args&&... args) {
    return builder_.create<OpTy>(loc_, std::forward<Args>(args)...);
  }

 private:
  XlaOp ConstantLiteral(const LiteralSlice& literal) override;

  StatusOr<XlaOp> ConvGeneralDilatedInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs, const Window& window,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config) override;

  StatusOr<XlaOp> FftInternal(const Shape& shape, XlaOp operand,
                              FftType fft_type,
                              absl::Span<const int64_t> fft_length) override;

  StatusOr<XlaOp> TriangularSolveInternal(
      const Shape& shape, XlaOp a, XlaOp b,
      TriangularSolveOptions options) override;

  StatusOr<XlaOp> CholeskyInternal(const Shape& shape, XlaOp a,
                                   bool lower) override;

  StatusOr<XlaOp> CustomCallInternal(
      const std::string& call_target_name, absl::Span<const XlaOp> operands,
      const XlaComputation* computation, const Shape& shape,
      const std::string& opaque,
      std::optional<absl::Span<const Shape>> operand_shapes_with_layout,
      bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, std::optional<Window> window,
      std::optional<ConvolutionDimensionNumbers> dnums,
      CustomCallSchedule schedule, CustomCallApiVersion api_version) override;

  StatusOr<XlaOp> ReduceInternal(
      const Shape& shape, absl::Span<const XlaOp> all_operands,
      const XlaComputation& computation,
      absl::Span<const int64_t> dimensions_to_reduce) override;

  StatusOr<XlaOp> ReduceWindowInternal(const Shape& shape, XlaOp operand,
                                       XlaOp init_value,
                                       const XlaComputation& computation,
                                       Window window) override;

  XlaOp Iota(const Shape& shape, int64_t iota_dimension) override;

  StatusOr<XlaOp> BitcastConvertTypeInternal(const Shape& shape,
                                             XlaOp operand) override;

  StatusOr<XlaOp> TransposeInternal(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64_t> permutation) override;

  StatusOr<XlaOp> RevInternal(const Shape& shape, XlaOp operand,
                              absl::Span<const int64_t> dimensions) override;

  StatusOr<XlaOp> SortInternal(const Shape& shape,
                               absl::Span<const XlaOp> operands,
                               const XlaComputation& comparator,
                               int64_t dimension, bool is_stable) override;

  StatusOr<XlaOp> WhileInternal(const Shape& shape,
                                const XlaComputation& condition,
                                const XlaComputation& body,
                                XlaOp init) override;

  StatusOr<XlaOp> ReducePrecisionInternal(const Shape& shape, XlaOp operand,
                                          const int exponent_bits,
                                          const int mantissa_bits) override;

  StatusOr<XlaOp> GatherInternal(
      const Shape& shape, XlaOp input, XlaOp start_indices,
      const GatherDimensionNumbers& dimension_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) override;

  StatusOr<XlaOp> ScatterInternal(
      const Shape& shape, absl::Span<const XlaOp> inputs, XlaOp scatter_indices,
      absl::Span<const XlaOp> updates, const XlaComputation& update_computation,
      const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
      bool unique_indices) override;

  StatusOr<XlaOp> SetDimensionSizeInternal(const Shape& shape, XlaOp operand,
                                           XlaOp val,
                                           int64_t dimension) override;

  StatusOr<XlaOp> RngOpInternal(RandomDistribution distribution,
                                absl::Span<const XlaOp> parameters,
                                const Shape& shape) override;
  StatusOr<XlaOp> RngBitGeneratorInternal(const Shape& full_result_shape,
                                          RandomAlgorithm algorithm,
                                          XlaOp initial_state) override;

  StatusOr<XlaOp> ReshapeInternal(const Shape& shape, XlaOp operand,
                                  int64_t inferred_dimension) override;

  StatusOr<XlaOp> DotGeneralInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs,
      const DotDimensionNumbers& dimension_number,
      const PrecisionConfig* precision_config) override;

  StatusOr<XlaOp> InDimBroadcast(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64_t> broadcast_dimensions) override;

  StatusOr<XlaOp> AddInstruction(HloInstructionProto&& instr, HloOpcode opcode,
                                 absl::Span<const XlaOp> operands) override;

  StatusOr<XlaOp> Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                          ComparisonDirection direction,
                          Comparison::Type type) override;

  XlaOp BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape, XlaOp lhs,
                            XlaOp rhs) override;

  StatusOr<XlaOp> AddOpWithShape(HloOpcode opcode, const Shape& shape,
                                 absl::Span<const XlaOp> operands) override;

  XlaOp CreateToken() override;

  StatusOr<XlaOp> InfeedWithTokenInternal(const Shape& infeed_instruction_shape,
                                          XlaOp token,
                                          const std::string& config) override;
  StatusOr<XlaOp> OutfeedWithTokenInternal(
      XlaOp operand, XlaOp token, const Shape& shape_with_layout,
      const std::string& outfeed_config) override;

  StatusOr<XlaOp> ConcatInDimInternal(const Shape& shape,
                                      absl::Span<const XlaOp> operands,
                                      int64_t dimension) override;

  StatusOr<XlaOp> GetTupleElementInternal(const Shape& shape, XlaOp tuple_data,
                                          int64_t index) override;

  StatusOr<XlaOp> SliceInternal(const Shape& shape, XlaOp operand,
                                absl::Span<const int64_t> start_indices,
                                absl::Span<const int64_t> limit_indices,
                                absl::Span<const int64_t> strides) override;

  StatusOr<XlaOp> DynamicSliceInternal(
      const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
      absl::Span<const int64_t> slice_sizes) override;

  StatusOr<XlaOp> DynamicUpdateSliceInternal(
      const Shape& shape, XlaOp operand, XlaOp update,
      absl::Span<const XlaOp> start_indices) override;

  StatusOr<XlaOp> PadInternal(const Shape& shape, XlaOp operand,
                              XlaOp padding_value,
                              const PaddingConfig& padding_config) override;

  StatusOr<XlaOp> TupleInternal(const Shape& shape,
                                absl::Span<const XlaOp> elements) override;

  // Creates HLO dialect op and returns the result as an XlaOp.
  StatusOr<XlaOp> CreateOp(
      const std::string& op_name, const Shape& shape,
      llvm::ArrayRef<XlaOp> operands,
      llvm::ArrayRef<mlir::NamedAttribute> attributes = {});

  Status ImportComputation(const HloModuleProto& computation,
                           mlir::Region* region,
                           bool flatten_region_arg_tuple = false);

  Status ImportComputation(const HloModuleProto& computation,
                           mlir::ModuleOp module);

  mlir::OpBuilder builder_;
  mlir::Location loc_;
  bool build_functions_;

  absl::flat_hash_map<int64_t, std::unique_ptr<Shape>> handle_to_shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
