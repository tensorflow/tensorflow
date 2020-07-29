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
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
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

  MlirHloBuilder(std::string name, mlir::OpBuilder builder, mlir::Location loc)
      : XlaBuilder(name), builder_(builder), loc_(loc) {}

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
      absl::Span<const int64> window_strides,
      absl::Span<const std::pair<int64, int64>> padding,
      absl::Span<const int64> lhs_dilation,
      absl::Span<const int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64 feature_group_count, int64 batch_group_count,
      const PrecisionConfig* precision_config) override;

  StatusOr<XlaOp> FftInternal(const Shape& shape, XlaOp operand,
                              FftType fft_type,
                              absl::Span<const int64> fft_length) override;

  StatusOr<XlaOp> CustomCallInternal(const string& call_target_name,
                                     absl::Span<const XlaOp> operands,
                                     const Shape& shape, const string& opaque,
                                     absl::optional<absl::Span<const Shape>>
                                         operand_shapes_with_layout) override;

  StatusOr<XlaOp> ReduceInternal(
      const Shape& shape, absl::Span<const XlaOp> all_operands,
      const XlaComputation& computation,
      absl::Span<const int64> dimensions_to_reduce) override;

  StatusOr<XlaOp> ReduceWindowInternal(const Shape& shape, XlaOp operand,
                                       XlaOp init_value,
                                       const XlaComputation& computation,
                                       Window window) override;

  XlaOp Iota(const Shape& shape, int64 iota_dimension) override;

  StatusOr<XlaOp> TransposeInternal(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64> permutation) override;

  StatusOr<XlaOp> RevInternal(const Shape& shape, XlaOp operand,
                              absl::Span<const int64> dimensions) override;

  StatusOr<XlaOp> GatherInternal(
      const Shape& shape, XlaOp input, XlaOp start_indices,
      const GatherDimensionNumbers& dimension_numbers,
      absl::Span<const int64> slice_sizes, bool indices_are_sorted) override;

  StatusOr<XlaOp> ScatterInternal(
      const Shape& shape, XlaOp input, XlaOp scatter_indices, XlaOp updates,
      const XlaComputation& update_computation,
      const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
      bool unique_indices) override;

  StatusOr<XlaOp> RngOpInternal(RandomDistribution distribution,
                                absl::Span<const XlaOp> parameters,
                                const Shape& shape) override;

  StatusOr<XlaOp> ReshapeInternal(const Shape& shape, XlaOp operand,
                                  int64 inferred_dimension) override;

  StatusOr<XlaOp> DotGeneralInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs,
      const DotDimensionNumbers& dimension_number,
      const PrecisionConfig* precision_config) override;

  StatusOr<XlaOp> InDimBroadcast(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64> broadcast_dimensions) override;

  StatusOr<XlaOp> Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                          ComparisonDirection direction) override;

  XlaOp BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape, XlaOp lhs,
                            XlaOp rhs) override;

  StatusOr<XlaOp> AddOpWithShape(HloOpcode opcode, const Shape& shape,
                                 absl::Span<const XlaOp> operands) override;

  XlaOp CreateToken() override;

  StatusOr<XlaOp> InfeedWithTokenInternal(const Shape& infeed_instruction_shape,
                                          XlaOp token,
                                          const string& config) override;
  StatusOr<XlaOp> OutfeedWithTokenInternal(
      XlaOp operand, XlaOp token, const Shape& shape_with_layout,
      const string& outfeed_config) override;

  StatusOr<XlaOp> ConcatInDimInternal(const Shape& shape,
                                      absl::Span<const XlaOp> operands,
                                      int64 dimension) override;

  StatusOr<XlaOp> GetTupleElementInternal(const Shape& shape, XlaOp tuple_data,
                                          int64 index) override;

  StatusOr<XlaOp> SliceInternal(const Shape& shape, XlaOp operand,
                                absl::Span<const int64> start_indices,
                                absl::Span<const int64> limit_indices,
                                absl::Span<const int64> strides) override;

  StatusOr<XlaOp> DynamicSliceInternal(
      const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
      absl::Span<const int64> slice_sizes) override;

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
                           mlir::Region* region);

  mlir::OpBuilder builder_;
  mlir::Location loc_;

  absl::flat_hash_map<int64, std::unique_ptr<Shape>> handle_to_shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
