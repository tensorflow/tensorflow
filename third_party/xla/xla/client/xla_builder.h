/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_CLIENT_XLA_BUILDER_H_
#define XLA_CLIENT_XLA_BUILDER_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/client/padding.h"
#include "xla/client/xla_computation.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dynamic_parameter_binding.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/bitmap.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

class XlaBuilder;
class XlaOp;
class HloInstruction;

namespace internal {

struct XlaBuilderFriend {
  static XlaOp BuildAddDependency(XlaBuilder* builder, XlaOp operand,
                                  XlaOp token, const Shape& shape);

  static std::pair<XlaOp, int64_t> BuildAsyncStart(
      XlaBuilder* builder, absl::Span<const XlaOp> operands,
      std::string execution_thread, int64_t group_id,
      const XlaComputation& called_computation, const Shape& shape);
  static std::pair<XlaOp, int64_t> BuildAsyncStart(
      XlaBuilder* builder, absl::Span<const XlaOp> operands,
      std::string execution_thread, const XlaComputation& called_computation,
      const Shape& shape);
  static XlaOp BuildAsyncUpdate(XlaBuilder* builder, XlaOp operands,
                                std::string execution_thread, int64_t group_id,
                                int64_t called_computation, const Shape& shape);
  static XlaOp BuildAsyncUpdate(XlaBuilder* builder, XlaOp operands,
                                std::string execution_thread,
                                int64_t called_computation, const Shape& shape);
  static XlaOp BuildAsyncDone(XlaBuilder* builder, XlaOp operands,
                              std::string execution_thread, int64_t group_id,
                              int64_t called_computation, const Shape& shape);
  static XlaOp BuildAsyncDone(XlaBuilder* builder, XlaOp operands,
                              std::string execution_thread,
                              int64_t called_computation, const Shape& shape);

  static XlaOp BuildAllGatherStart(
      XlaBuilder* builder, XlaOp operand, int64_t all_gather_dimension,
      int64_t shard_count, absl::Span<const ReplicaGroup> replica_groups = {},
      const std::optional<ChannelHandle>& channel_id = std::nullopt,
      const std::optional<Layout>& layout = std::nullopt,
      std::optional<bool> use_global_device_ids = std::nullopt);
  static XlaOp BuildAllGatherDone(XlaBuilder* builder, XlaOp operands,
                                  const Shape& shape);

  static XlaOp BuildAllReduceStart(
      XlaBuilder* builder, XlaOp operand, const XlaComputation& computation,
      absl::Span<const ReplicaGroup> replica_groups = {},
      const std::optional<ChannelHandle>& channel_id = std::nullopt,
      const std::optional<Shape>& layout = std::nullopt,
      std::optional<bool> use_global_device_ids = std::nullopt);
  static XlaOp BuildAllReduceDone(XlaBuilder* builder, XlaOp operands,
                                  const Shape& shape);

  static XlaOp BuildCollectivePermuteStart(
      XlaBuilder* builder, XlaOp operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<ChannelHandle>& channel_id = std::nullopt);
  static XlaOp BuildCollectivePermuteDone(XlaBuilder* builder, XlaOp operands,
                                          const Shape& shape);

  static XlaOp BuildCopyStart(
      XlaBuilder* builder, XlaOp operand,
      std::optional<int> cross_program_prefetch_index = std::nullopt);
  static XlaOp BuildCopyDone(XlaBuilder* builder, XlaOp operand,
                             const Shape& shape);

  static XlaOp BuildFusion(
      XlaBuilder* builder, absl::Span<const XlaOp> operands,
      absl::string_view fusion_kind, const XlaComputation& fused_computation,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing = {});

  static XlaOp BuildBitcast(XlaBuilder* builder, XlaOp operand,
                            const Shape& shape);

  static XlaOp BuildPartitionId(XlaBuilder* builder, const Shape& shape);

  static XlaOp BuildSend(XlaBuilder* builder, XlaOp operand, XlaOp token,
                         const ChannelHandle& handle, bool is_host_transfer);
  static XlaOp BuildSendDone(XlaBuilder* builder, XlaOp operand,
                             const ChannelHandle& handle,
                             bool is_host_transfer);

  static XlaOp BuildRecv(XlaBuilder* builder, XlaOp token, const Shape& shape,
                         const ChannelHandle& handle, bool is_host_transfer);
  static XlaOp BuildRecvDone(XlaBuilder* builder, XlaOp token,
                             const Shape& shape, const ChannelHandle& handle,
                             bool is_host_transfer);

  static XlaOp BuildDomain(XlaBuilder* builder, XlaOp operand, OpSharding entry,
                           OpSharding exit, const Shape& shape);

  static XlaOp BuildRngGetAndUpdateState(XlaBuilder* builder, int64_t delta,
                                         const Shape& shape);

  static HloInstructionProto* GetInstruction(XlaOp op);
  static HloInstructionProto* GetInstructionByHandle(XlaBuilder* builder,
                                                     int64_t handle);
};

}  // namespace internal

// This represents an instruction that has been enqueued using the XlaBuilder.
// This is used to pass to subsequent computations that depends upon the
// instruction as an operand.
class XlaOp {
 public:
  XlaOp() : handle_(-1), builder_(nullptr) {
    static_assert(std::is_trivially_destructible<XlaOp>::value,
                  "XlaOp should be trivially destructible");
  }
  ~XlaOp() = default;

  XlaOp(const XlaOp& other) = default;
  XlaOp& operator=(const XlaOp& other) = default;

  // Precondition: !IsUninitialized().
  //
  // It's very common to do foo.builder()->bar().  Without this precondition, if
  // foo.builder() is null, the call to bar will segfault at some point possibly
  // deep in the callstack when we finally dereference `this`.  The precondition
  // lets us avoid this tricky-to-debug problem.
  XlaBuilder* builder() const {
    CHECK(builder_ != nullptr);
    return builder_;
  }

  // Returns true if the XlaOp represents valid, non-erroneous value.
  bool valid() const { return handle_ >= 0; }

  // Returns true if the XlaOp was created by the XlaOp() constructor and
  // not returned by a builder.
  bool IsUninitialized() const { return builder_ == nullptr; }

  bool IsIdenticalTo(XlaOp rhs) const {
    return handle_ == rhs.handle_ && builder_ == rhs.builder_;
  }

  friend std::ostream& operator<<(std::ostream& out, XlaOp op) {
    out << op.handle();
    return out;
  }

 private:
  explicit XlaOp(XlaBuilder* builder) : handle_(-1), builder_(builder) {}
  XlaOp(int64_t handle, XlaBuilder* builder)
      : handle_(handle), builder_(builder) {}

  int64_t handle() const { return handle_; }

  friend class XlaBuilder;
  friend class ValueInference;
  friend struct internal::XlaBuilderFriend;

  // < 0 means "invalid handle".
  int64_t handle_;

  // Not owned. Non-null for any handle returned by XlaBuilder, even if the
  // handle is invalid.
  XlaBuilder* builder_;
};

// Arithmetic operator overloads for the XlaOp type.
XlaOp operator-(XlaOp x);
XlaOp operator+(XlaOp x, XlaOp y);
XlaOp operator-(XlaOp x, XlaOp y);
XlaOp operator*(XlaOp x, XlaOp y);
XlaOp operator/(XlaOp x, XlaOp y);
XlaOp operator%(XlaOp x, XlaOp y);

// Bitwise operator overloads for the XlaOp type.
XlaOp operator~(XlaOp x);
XlaOp operator&(XlaOp x, XlaOp y);
XlaOp operator|(XlaOp x, XlaOp y);
XlaOp operator^(XlaOp x, XlaOp y);
XlaOp operator<<(XlaOp x, XlaOp y);
// Performs a right arithmetic shift if 'x' is a signed type, otherwise performs
// a right logical shift.
XlaOp operator>>(XlaOp x, XlaOp y);

// We don't overload the relational operators (==, !=, <, <=, >, >=) because the
// semantics might be surprising since their result types are usually 'bool'.
// Further programmers may expect == to be a structural equality.
// We also choose not to overload any of the mutating operators (e.g., +=, -=)
// because the semantics might be misleading — XLA computations are immutable.

// A convenient interface for building up computations.
//
// Thread-compatible.
class XlaBuilder {
 public:
  // computation_name: name to use for the built computation.
  explicit XlaBuilder(const std::string& computation_name);

  XlaBuilder(const XlaBuilder&) = delete;
  XlaBuilder& operator=(const XlaBuilder&) = delete;

  virtual ~XlaBuilder();

  // Returns the computation name.
  const std::string& name() const { return name_; }

  // Sets OpMetadata that will be added to all instructions until cleared.
  //
  // OpMetadata is often applied to a series of XLA HLO instructions. As a
  // result, OpMetadata is set on the computation builder. All subsequent
  // instructions generated via this computation builder will have the same
  // OpMetadata attached until a call to ClearOpMetadata.
  void SetOpMetadata(OpMetadata metadata) { metadata_ = std::move(metadata); }

  // Swaps the passed op metadata with the ones currently set.
  //
  // Returns the old op metadata.
  OpMetadata SwapOpMetadata(OpMetadata metadata) {
    OpMetadata old_metadata = std::move(metadata_);
    metadata_ = std::move(metadata);
    return old_metadata;
  }

  // Similar to SetOpMetadata, but only set the metadata for the next op.
  void SetOneShotOpMetadata(OpMetadata metadata) {
    one_shot_metadata_ = std::move(metadata);
  }

  // Clears the HloMetadata state.
  void ClearOpMetadata() { metadata_.Clear(); }

  // Sets an OpSharding that will be attached to all instructions until cleared.
  void SetSharding(const OpSharding& sharding) { sharding_ = sharding; }

  // Sets the FrontendAttributes that will be added to all instructions until
  // cleared.
  //
  // FrontendAttributes are often applied to a series of XLA HLO instructions.
  // As a result they are set on the computation builder and all the
  // instructions generated via the computation builder will have the same
  // frontend attributes attached to them.
  virtual void SetFrontendAttributes(
      const FrontendAttributes& frontend_attributes) {
    frontend_attributes_ = frontend_attributes;
  }

  // Swap the passed FrontendAttributes with the ones currently set.
  //
  // Return the old attributes.
  FrontendAttributes SwapFrontendAttributes(
      const FrontendAttributes& frontend_attributes) {
    FrontendAttributes old_attributes = std::move(frontend_attributes_);
    frontend_attributes_ = frontend_attributes;
    return old_attributes;
  }

  // Returns the FrontendAttributes that will be attached to all instructions.
  const FrontendAttributes& frontend_attributes() const {
    return frontend_attributes_;
  }

  // Clears all the frontend attributes.
  void ClearFrontendAttributes() { frontend_attributes_.Clear(); }

  // Clears the sharding. Ops will be sharded according to the default placement
  // policy.
  void ClearSharding() { sharding_ = std::nullopt; }

  // Returns the OpSharding that will be attached to all instructions.
  const std::optional<OpSharding>& sharding() const { return sharding_; }

  // Sets the builder to a mode where it will die immediately when an error is
  // encountered, rather than producing it in a deferred fashion when Build() is
  // called (which is the default).
  void set_die_immediately_on_error(bool enabled) {
    die_immediately_on_error_ = enabled;
  }

  // Default dimension numbers used for a 2D convolution.
  static constexpr int64_t kConvBatchDimension = 0;
  static constexpr int64_t kConvFeatureDimension = 1;
  static constexpr int64_t kConvFirstSpatialDimension = 2;
  static constexpr int64_t kConvSecondSpatialDimension = 3;
  static constexpr int64_t kConvKernelOutputDimension = 0;
  static constexpr int64_t kConvKernelInputDimension = 1;
  static constexpr int64_t kConvKernelFirstSpatialDimension = 2;
  static constexpr int64_t kConvKernelSecondSpatialDimension = 3;

  // Creates a default ConvolutionDimensionNumbers. For a 2D convolution, for
  // the input operand {batch, feature, height, width} = {0, 1, 2, 3} and for
  // the kernel operand
  // {output_feature, input_feature, height, width} = {0, 1, 2, 3}.
  static ConvolutionDimensionNumbers CreateDefaultConvDimensionNumbers(
      int num_spatial_dims = 2);

  // Returns an error if the convolution dimension numbers have conflicts.
  static Status Validate(const ConvolutionDimensionNumbers& dnum);

  // Returns a new XlaBuilder whose resultant Computation is used only by this
  // XlaBuilder. The sub-XlaBuilder has the same die_immediately_on_error
  // behavior as the parent.
  std::unique_ptr<XlaBuilder> CreateSubBuilder(
      const std::string& computation_name);

  // Builds the computation with the requested operations, or returns a non-ok
  // status. Note that all ops that have been enqueued will be moved to the
  // computation being returned. The root of the computation will be the last
  // added operation.
  //
  // `remove_dynamic_dimensions` tells the builder whether to remove the
  // dynamic dimensions information in all ops.
  //
  // TODO(b/121223198): Delete `remove_dynamic_dimensions` and keeps the
  // dynamic dimensions information when XLA backend can handle dynamic
  // dimensions.
  StatusOr<XlaComputation> Build(bool remove_dynamic_dimensions = false);

  // Overload of Build which specifies a particular root instruction for the
  // computation.
  StatusOr<XlaComputation> Build(XlaOp root,
                                 bool remove_dynamic_dimensions = false);

  // Builds the computation with the requested operations, or notes an error in
  // the parent XlaBuilder and returns an empty computation if building failed.
  // This function is intended to be used where the returned XlaComputation is
  // only used by the parent XlaBuilder and hence further operation on the
  // returned XlaComputation will simply be error'ed out if an error occurred
  // while building this computation. If the built computation is to be used by
  // a XlaBuilder other than the parent XlaBuilder then Build() should be used
  // instead.
  XlaComputation BuildAndNoteError();

  // Returns a subgraph that roots on the given root. If the root is not a
  // compile-time constant (see `IsConstant`), returns an error.
  //
  // This will copy the needed ops/computations to the subgraph.
  StatusOr<XlaComputation> BuildConstantSubGraph(
      XlaOp root_op, bool dynamic_dimension_is_minus_one = false);

  // Returns the first error that was encountered while building the
  // computation. When an error is encountered, by default we return a vacuous
  // XlaOp and inform the user of the error that occurred while
  // building the computation when they make a final call to Build().
  //
  // See also set_die_immediately_on_error().
  Status first_error() const { return first_error_; }

  // Returns the current status of the builder, complete with the stack trace
  // information.
  Status GetCurrentStatus() const;

  // Returns the shape of the given op.
  StatusOr<Shape> GetShape(XlaOp op) const;

  // Returns the shape of the given op.
  virtual StatusOr<const Shape*> GetShapePtr(XlaOp op) const;

  // Returns the (inferred) result for the current computation's shape. This
  // assumes the root instruction is the last added instruction.
  StatusOr<ProgramShape> GetProgramShape() const;

  // Returns the (inferred) result for the current computation's shape using the
  // given operation as the root.
  StatusOr<ProgramShape> GetProgramShape(XlaOp root) const;

  // Reports an error to the builder, by
  // * storing it internally and capturing a backtrace if it's the first error
  //   (this deferred value will be produced on the call to
  //    Build()/GetShape()/...)
  // * dying if die_immediately_on_error_ is true.
  // Returns an XlaOp with an invalid handle but a valid builder. This value can
  // be returned in place of a value in APIs that return an XlaOp.
  XlaOp ReportError(const Status& error);

  // A helper function that converts a StatusOr<XlaOp> into an XlaOp.
  // If the Status was an error, reports the error to builder and returns an
  // invalid XlaOp handle.
  XlaOp ReportErrorOrReturn(const StatusOr<XlaOp>& op);

  // A helper function that runs a function that returns a StatusOr<XlaOp> and
  // returns an XlaOp.
  XlaOp ReportErrorOrReturn(absl::FunctionRef<StatusOr<XlaOp>()> op_creator);

  // Returns true if 'operand' is a compile-time constant. A compile-time
  // constant does not depend on any parameters, or on stateful operators such
  // as `RngNormal` or `Infeed`.
  //
  // This tests whether a computation is a compile-time constant without
  // evaluating the computation.
  StatusOr<bool> IsConstant(XlaOp operand) const;

  // Adds a new input/output alias. Since the input/output shape information are
  // not available until the computation is built, any eventual error in the
  // arguments of this API will be detected only at computation Build() time.
  //
  // Note: Except when 'must-alias' is true, alias is assumed to be 'may-alias'
  // and only donated buffer at runtime will be aliased with output. If a buffer
  // is not donated at runtime, a copy will be inserted by XLA to prevent buffer
  // clobbering.
  void SetUpAlias(const ShapeIndex& output_index, int64_t param_number,
                  const ShapeIndex& param_index,
                  HloInputOutputAliasConfig::AliasKind kind =
                      HloInputOutputAliasConfig::AliasKind::kMayAlias) {
    input_output_aliases_.push_back(
        {output_index, param_number, param_index, kind});
  }

  // Describes an input/output alias as inserted by the SetUpAlias() API.
  struct InputOutputAlias {
    // Specifies the index of the aliased buffer in the result tuple.
    ShapeIndex output_index;
    // Specifies the parameter containing the buffer to be aliased.
    int64_t param_number;
    // Specifies the index of the aliased buffer in the parameter.
    ShapeIndex param_index;
    // Specifies if the alias is a must alias or may alias.
    HloInputOutputAliasConfig::AliasKind kind;
  };

  // Adds a new buffer donor. The donated buffer may be paired with any valid
  // output. On the contrary, the buffer aliasing bonds the input output pair.
  // The input can only donate the buffer to the paired output.
  void AddBufferDonor(int64_t param_number, const ShapeIndex& param_index) {
    buffer_donors_.insert({param_number, param_index});
  }

  // Looks up the HloInstruction and sets the frontend attribute "attribute" to
  // "value".
  //
  // If the attribute already existed then its value is updated.
  //
  // Note: the attribute is only added to the HloInstruction, not to the
  // builder.
  Status SetInstructionFrontendAttribute(XlaOp op, std::string attribute,
                                         std::string value);

  // Returns shapes for the operands.
  StatusOr<std::vector<Shape>> GetOperandShapes(
      absl::Span<const XlaOp> operands) const;

  // Converts the op to string for the ease of debugging.
  std::string OpToString(XlaOp op) const;

 private:
  void ToStringHelper(std::string* out, int ident, int64_t op_handle) const;

  // Build helper which takes the id of the root operation..
  StatusOr<XlaComputation> Build(int64_t root_id,
                                 bool remove_dynamic_dimensions);

  // Description for the methods below can be found in the corresponding public
  // functions section in this file.

  XlaOp Parameter(int64_t parameter_number, const Shape& shape,
                  const std::string& name,
                  const std::vector<bool>& replicated_at_leaf_buffers);
  XlaOp Parameter(int64_t parameter_number, const Shape& shape,
                  const std::string& name) {
    std::vector<bool> empty_bools;
    return Parameter(parameter_number, shape, name, empty_bools);
  }

  virtual XlaOp ConstantLiteral(const LiteralSlice& literal);

  XlaOp Broadcast(XlaOp operand, absl::Span<const int64_t> broadcast_sizes);

  XlaOp BroadcastInDim(XlaOp operand, absl::Span<const int64_t> out_dim_size,
                       absl::Span<const int64_t> broadcast_dimensions);

  XlaOp Pad(XlaOp operand, XlaOp padding_value,
            const PaddingConfig& padding_config);
  XlaOp PadInDim(XlaOp operand, XlaOp padding_value, int64_t dimno,
                 int64_t pad_lo, int64_t pad_hi);

  virtual StatusOr<XlaOp> PadInternal(const Shape& shape, XlaOp operand,
                                      XlaOp padding_value,
                                      const PaddingConfig& padding_config);

  XlaOp Reshape(XlaOp operand, absl::Span<const int64_t> dimensions,
                absl::Span<const int64_t> new_sizes,
                int64_t inferred_dimension = -1);

  XlaOp Reshape(XlaOp operand, absl::Span<const int64_t> new_sizes,
                int64_t inferred_dimension = -1);

  XlaOp Reshape(const Shape& shape, XlaOp operand,
                int64_t inferred_dimension = -1);

  XlaOp DynamicReshape(XlaOp operand, absl::Span<const XlaOp> dim_sizes,
                       absl::Span<const int64_t> new_size_bounds,
                       const std::vector<bool>& dims_are_dynamic);

  XlaOp Collapse(XlaOp operand, absl::Span<const int64_t> dimensions);

  XlaOp Slice(XlaOp operand, absl::Span<const int64_t> start_indices,
              absl::Span<const int64_t> limit_indices,
              absl::Span<const int64_t> strides);
  virtual StatusOr<XlaOp> SliceInternal(const Shape& shape, XlaOp operand,
                                        absl::Span<const int64_t> start_indices,
                                        absl::Span<const int64_t> limit_indices,
                                        absl::Span<const int64_t> strides);
  virtual XlaOp SliceInDim(XlaOp operand, int64_t start_index,
                           int64_t limit_index, int64_t stride, int64_t dimno);

  XlaOp DynamicSlice(XlaOp operand, absl::Span<const XlaOp> start_indices,
                     absl::Span<const int64_t> slice_sizes);
  virtual StatusOr<XlaOp> DynamicSliceInternal(
      const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
      absl::Span<const int64_t> slice_sizes);

  XlaOp DynamicUpdateSlice(XlaOp operand, XlaOp update,
                           absl::Span<const XlaOp> start_indices);
  virtual StatusOr<XlaOp> DynamicUpdateSliceInternal(
      const Shape& shape, XlaOp operand, XlaOp update,
      absl::Span<const XlaOp> start_indices);

  XlaOp ConcatInDim(absl::Span<const XlaOp> operands, int64_t dimension);
  virtual StatusOr<XlaOp> ConcatInDimInternal(const Shape& shape,
                                              absl::Span<const XlaOp> operands,
                                              int64_t dimension);

  XlaOp Select(XlaOp pred, XlaOp on_true, XlaOp on_false);

  XlaOp Tuple(absl::Span<const XlaOp> elements);
  virtual StatusOr<XlaOp> TupleInternal(const Shape& shape,
                                        absl::Span<const XlaOp> elements);

  XlaOp GetTupleElement(XlaOp tuple_data, int64_t index);
  virtual StatusOr<XlaOp> GetTupleElementInternal(const Shape& shape,
                                                  XlaOp tuple_data,
                                                  int64_t index);

  XlaOp Dot(XlaOp lhs, XlaOp rhs,
            const PrecisionConfig* precision_config = nullptr,
            std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp DotGeneral(
      XlaOp lhs, XlaOp rhs, const DotDimensionNumbers& dimension_numbers,
      const PrecisionConfig* precision_config = nullptr,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp Conv(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      Padding padding, int64_t feature_group_count = 1,
      int64_t batch_group_count = 1,
      const PrecisionConfig* precision_config = nullptr,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp ConvWithGeneralPadding(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      int64_t feature_group_count = 1, int64_t batch_group_count = 1,
      const PrecisionConfig* precision_config = nullptr,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp ConvWithGeneralDimensions(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count = 1, int64_t batch_group_count = 1,
      const PrecisionConfig* precision_config = nullptr,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp ConvGeneral(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count = 1, int64_t batch_group_count = 1,
      const PrecisionConfig* precision_config = nullptr,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp ConvGeneralDilated(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count = 1, int64_t batch_group_count = 1,
      const PrecisionConfig* precision_config = nullptr,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt,
      std::optional<std::vector<bool>> window_reversal = std::nullopt);

  XlaOp DynamicConvForward(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp DynamicConvInputGrad(
      XlaOp input_sizes, XlaOp lhs, XlaOp rhs,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  XlaOp DynamicConvKernelGrad(
      XlaOp activations, XlaOp gradients,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  StatusOr<HloInstructionProto> DynamicConvInstruction(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type = std::nullopt);

  virtual StatusOr<XlaOp> ConvGeneralDilatedInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs, const Window& window,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config);

  XlaOp Fft(XlaOp operand, FftType fft_type,
            absl::Span<const int64_t> fft_length);
  virtual StatusOr<XlaOp> FftInternal(const Shape& shape, XlaOp operand,
                                      FftType fft_type,
                                      absl::Span<const int64_t> fft_length);

  virtual StatusOr<XlaOp> TriangularSolveInternal(
      const Shape& shape, XlaOp a, XlaOp b, TriangularSolveOptions options);

  virtual StatusOr<XlaOp> CholeskyInternal(const Shape& shape, XlaOp a,
                                           bool lower);

  XlaOp Infeed(const Shape& shape, const std::string& config = "");
  XlaOp InfeedWithToken(XlaOp token, const Shape& shape,
                        const std::string& config);
  virtual StatusOr<XlaOp> InfeedWithTokenInternal(
      const Shape& infeed_instruction_shape, XlaOp token,
      const std::string& config);

  void Outfeed(XlaOp operand, const Shape& shape_with_layout,
               const std::string& outfeed_config);
  XlaOp OutfeedWithToken(XlaOp operand, XlaOp token,
                         const Shape& shape_with_layout,
                         const std::string& outfeed_config);
  virtual StatusOr<XlaOp> OutfeedWithTokenInternal(
      XlaOp operand, XlaOp token, const Shape& shape_with_layout,
      const std::string& outfeed_config);
  XlaOp Call(const XlaComputation& computation,
             absl::Span<const XlaOp> operands);

  XlaOp CustomCall(
      const std::string& call_target_name, absl::Span<const XlaOp> operands,
      const Shape& shape_with_layout, const std::string& opaque,
      std::optional<absl::Span<const Shape>> operand_shapes_with_layout,
      bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, std::optional<Window> window,
      std::optional<ConvolutionDimensionNumbers> dnums,
      CustomCallSchedule schedule, CustomCallApiVersion api_version);

  // Internal version of CustomCall without computation that doesn't do op
  // specific error handling and expects arguments to be legal. CustomCall
  // method above calls this method after error handling.
  virtual StatusOr<XlaOp> CustomCallInternal(
      const std::string& call_target_name, absl::Span<const XlaOp> operands,
      const XlaComputation* computation, const Shape& shape_with_layout,
      const std::string& opaque,
      std::optional<absl::Span<const Shape>> operand_shapes_with_layout,
      bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, std::optional<Window> window,
      std::optional<ConvolutionDimensionNumbers> dnums,
      CustomCallSchedule schedule, CustomCallApiVersion api_version);

  // TODO(b/239474321) Remove this overload as it has simply led to code
  // duplication.
  XlaOp CustomCall(
      const std::string& call_target_name, absl::Span<const XlaOp> operands,
      const XlaComputation& computation, const Shape& shape_with_layout,
      const std::string& opaque,
      std::optional<absl::Span<const Shape>> operand_shapes_with_layout,
      bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, CustomCallSchedule schedule,
      CustomCallApiVersion api_version);

  XlaOp OptimizationBarrier(XlaOp operand);

  XlaOp Reduce(XlaOp operand, XlaOp init_value,
               const XlaComputation& computation,
               absl::Span<const int64_t> dimensions_to_reduce);

  XlaOp Reduce(absl::Span<const XlaOp> operands,
               absl::Span<const XlaOp> init_values,
               const XlaComputation& computation,
               absl::Span<const int64_t> dimensions_to_reduce);

  virtual StatusOr<XlaOp> ReduceInternal(
      const Shape& shape, absl::Span<const XlaOp> all_operands,
      const XlaComputation& computation,
      absl::Span<const int64_t> dimensions_to_reduce);

  XlaOp ReduceAll(XlaOp operand, XlaOp init_value,
                  const XlaComputation& computation);

  XlaOp ReduceWindow(XlaOp operand, XlaOp init_value,
                     const XlaComputation& computation,
                     absl::Span<const int64_t> window_dimensions,
                     absl::Span<const int64_t> window_strides, Padding padding);

  XlaOp ReduceWindow(absl::Span<const XlaOp> operands,
                     absl::Span<const XlaOp> init_values,
                     const XlaComputation& computation,
                     absl::Span<const int64_t> window_dimensions,
                     absl::Span<const int64_t> window_strides, Padding padding);

  XlaOp ReduceWindowWithGeneralPadding(
      absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
      const XlaComputation& computation,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const int64_t> base_dilations,
      absl::Span<const int64_t> window_dilations,
      absl::Span<const std::pair<int64_t, int64_t>> padding);
  StatusOr<HloInstructionProto> ReduceWindowInternal(
      absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
      const XlaComputation& computation,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const int64_t> base_dilations,
      absl::Span<const int64_t> window_dilations,
      absl::Span<const std::pair<int64_t, int64_t>> padding);
  virtual StatusOr<XlaOp> ReduceWindowInternal(
      const Shape& shape, XlaOp operand, XlaOp init_value,
      const XlaComputation& computation, Window window);
  XlaOp CrossReplicaSum(XlaOp operand,
                        absl::Span<const ReplicaGroup> replica_groups = {});

  XlaOp AllGather(XlaOp operand, int64_t all_gather_dimension,
                  int64_t shard_count,
                  absl::Span<const ReplicaGroup> replica_groups = {},
                  const std::optional<ChannelHandle>& channel_id = std::nullopt,
                  const std::optional<Layout>& layout = std::nullopt,
                  std::optional<bool> use_global_device_ids = std::nullopt);

  XlaOp AllReduce(XlaOp operand, const XlaComputation& computation,
                  absl::Span<const ReplicaGroup> replica_groups = {},
                  const std::optional<ChannelHandle>& channel_id = std::nullopt,
                  const std::optional<Shape>& shape_with_layout = std::nullopt,
                  std::optional<bool> use_global_device_ids = std::nullopt);

  XlaOp ReduceScatter(
      XlaOp operand, const XlaComputation& computation,
      int64_t scatter_dimension, int64_t shard_count,
      absl::Span<const ReplicaGroup> replica_groups = {},
      const std::optional<ChannelHandle>& channel_id = std::nullopt,
      const std::optional<Layout>& layout = std::nullopt,
      std::optional<bool> use_global_device_ids = std::nullopt);

  XlaOp AllToAll(XlaOp operand, int64_t split_dimension,
                 int64_t concat_dimension, int64_t split_count,
                 absl::Span<const ReplicaGroup> replica_groups,
                 const std::optional<Layout>& layout = std::nullopt,
                 const std::optional<ChannelHandle>& channel_id = std::nullopt);

  XlaOp AllToAllTuple(
      absl::Span<const XlaOp> operands,
      absl::Span<const ReplicaGroup> replica_groups,
      const std::optional<Layout>& layout,
      const std::optional<ChannelHandle>& channel_id = std::nullopt);

  XlaOp AllToAllTuple(
      XlaOp operand, int64_t split_dimension, int64_t concat_dimension,
      int64_t split_count, absl::Span<const ReplicaGroup> replica_groups,
      const std::optional<Layout>& layout,
      const std::optional<ChannelHandle>& channel_id = std::nullopt);

  XlaOp CollectivePermute(
      XlaOp operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<ChannelHandle>& channel_id = std::nullopt);

  XlaOp ReplicaId();

  XlaOp SelectAndScatter(XlaOp operand, const XlaComputation& select,
                         absl::Span<const int64_t> window_dimensions,
                         absl::Span<const int64_t> window_strides,
                         Padding padding, XlaOp source, XlaOp init_value,
                         const XlaComputation& scatter);

  XlaOp SelectAndScatterWithGeneralPadding(
      XlaOp operand, const XlaComputation& select,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding, XlaOp source,
      XlaOp init_value, const XlaComputation& scatter);

  StatusOr<HloInstructionProto> SelectAndScatterInternal(
      XlaOp operand, const XlaComputation& select,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding, XlaOp source,
      XlaOp init_value, const XlaComputation& scatter);

  virtual XlaOp Iota(const Shape& shape, int64_t iota_dimension);

  XlaOp Iota(PrimitiveType type, int64_t size);

  XlaOp ConvertElementType(XlaOp operand, PrimitiveType new_element_type);

  XlaOp BitcastConvertType(XlaOp operand, PrimitiveType new_element_type);
  virtual StatusOr<XlaOp> BitcastConvertTypeInternal(const Shape& shape,
                                                     XlaOp operand);

  XlaOp StochasticConvertType(XlaOp operand, XlaOp random,
                              PrimitiveType new_element_type);

  XlaOp Transpose(XlaOp operand, absl::Span<const int64_t> permutation);
  virtual StatusOr<XlaOp> TransposeInternal(
      const Shape& shape, XlaOp operand, absl::Span<const int64_t> permutation);

  XlaOp Rev(XlaOp operand, absl::Span<const int64_t> dimensions);
  virtual StatusOr<XlaOp> RevInternal(const Shape& shape, XlaOp operand,
                                      absl::Span<const int64_t> dimensions);

  XlaOp Sort(absl::Span<const XlaOp> operands, const XlaComputation& comparator,
             int64_t dimension = -1, bool is_stable = false);
  virtual StatusOr<XlaOp> SortInternal(const Shape& shape,
                                       absl::Span<const XlaOp> operands,
                                       const XlaComputation& comparator,
                                       int64_t dimension, bool is_stable);

  XlaOp TopK(XlaOp operand, int64_t k, bool largest);
  virtual StatusOr<XlaOp> TopKInternal(const Shape& shape, XlaOp operand,
                                       int64_t k, bool largest);

  XlaOp Clamp(XlaOp min, XlaOp operand, XlaOp max);

  XlaOp Map(absl::Span<const XlaOp> operands, const XlaComputation& computation,
            absl::Span<const int64_t> dimensions,
            absl::Span<const XlaOp> static_operands = {});

  XlaOp RngNormal(XlaOp mu, XlaOp sigma, const Shape& shape);

  XlaOp RngUniform(XlaOp a, XlaOp b, const Shape& shape);

  XlaOp RngBitGenerator(RandomAlgorithm algorithm, XlaOp initial_state,
                        const Shape& shape);
  // Internal variant for the op with the full result shape containing both data
  // and state shape as a tuple.
  virtual StatusOr<XlaOp> RngBitGeneratorInternal(
      const Shape& full_result_shape, RandomAlgorithm algorithm,
      XlaOp initial_state);

  XlaOp While(const XlaComputation& condition, const XlaComputation& body,
              XlaOp init);
  virtual StatusOr<XlaOp> WhileInternal(const Shape& shape,
                                        const XlaComputation& condition,
                                        const XlaComputation& body, XlaOp init);

  XlaOp Conditional(XlaOp predicate, XlaOp true_operand,
                    const XlaComputation& true_computation, XlaOp false_operand,
                    const XlaComputation& false_computation);

  XlaOp Conditional(XlaOp branch_index,
                    absl::Span<const XlaComputation* const> branch_computations,
                    absl::Span<const XlaOp> branch_operands);

  XlaOp ReducePrecision(XlaOp operand, int exponent_bits, int mantissa_bits);
  virtual StatusOr<XlaOp> ReducePrecisionInternal(const Shape& shape,
                                                  XlaOp operand,
                                                  int exponent_bits,
                                                  int mantissa_bits);

  XlaOp Gather(XlaOp input, XlaOp start_indices,
               const GatherDimensionNumbers& dimension_numbers,
               absl::Span<const int64_t> slice_sizes,
               bool indices_are_sorted = false);

  virtual StatusOr<XlaOp> GatherInternal(
      const Shape& shape, XlaOp input, XlaOp start_indices,
      const GatherDimensionNumbers& dimension_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted);

  XlaOp Scatter(XlaOp input, XlaOp scatter_indices, XlaOp updates,
                const XlaComputation& update_computation,
                const ScatterDimensionNumbers& dimension_numbers,
                bool indices_are_sorted = false, bool unique_indices = false);
  XlaOp Scatter(absl::Span<const XlaOp> inputs, XlaOp scatter_indices,
                absl::Span<const XlaOp> updates,
                const XlaComputation& update_computation,
                const ScatterDimensionNumbers& dimension_numbers,
                bool indices_are_sorted = false, bool unique_indices = false);

  virtual StatusOr<XlaOp> ScatterInternal(
      const Shape& shape, absl::Span<const XlaOp> inputs, XlaOp scatter_indices,
      absl::Span<const XlaOp> updates, const XlaComputation& update_computation,
      const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
      bool unique_indices);

  void Send(XlaOp operand, const ChannelHandle& handle);
  XlaOp SendWithToken(XlaOp operand, XlaOp token, const ChannelHandle& handle);

  XlaOp SendToHost(XlaOp operand, XlaOp token, const Shape& shape_with_layout,
                   const ChannelHandle& handle);

  XlaOp RecvFromHost(XlaOp token, const Shape& shape,
                     const ChannelHandle& handle);

  virtual XlaOp CreateToken();

  XlaOp AfterAll(absl::Span<const XlaOp> tokens);

  XlaOp Recv(const Shape& shape, const ChannelHandle& handle);
  XlaOp RecvWithToken(XlaOp token, const Shape& shape,
                      const ChannelHandle& handle);

  XlaOp BatchNormTraining(XlaOp operand, XlaOp scale, XlaOp offset,
                          float epsilon, int64_t feature_index);

  XlaOp BatchNormInference(XlaOp operand, XlaOp scale, XlaOp offset, XlaOp mean,
                           XlaOp variance, float epsilon,
                           int64_t feature_index);

  XlaOp BatchNormGrad(XlaOp operand, XlaOp scale, XlaOp batch_mean,
                      XlaOp batch_var, XlaOp grad_output, float epsilon,
                      int64_t feature_index);

  XlaOp GetDimensionSize(XlaOp operand, int64_t dimension);

  XlaOp SetDimensionSize(XlaOp operand, XlaOp val, int64_t dimension);

  virtual StatusOr<XlaOp> SetDimensionSizeInternal(const Shape& shape,
                                                   XlaOp operand, XlaOp val,
                                                   int64_t dimension);

  XlaOp RemoveDynamicDimension(XlaOp operand, int64_t dimension);

  virtual StatusOr<XlaOp> AddInstruction(HloInstructionProto&& instr,
                                         HloOpcode opcode,
                                         absl::Span<const XlaOp> operands);
  StatusOr<XlaOp> AddInstruction(HloInstructionProto&& instr,
                                 HloOpcode opcode) {
    return AddInstruction(std::move(instr), opcode, /*operands=*/{});
  }

  void AddCalledComputation(const XlaComputation& computation,
                            HloInstructionProto* instr);

  StatusOr<const HloInstructionProto*> LookUpInstruction(XlaOp op) const;
  StatusOr<const HloInstructionProto*> LookUpInstructionByHandle(
      int64_t handle) const;
  StatusOr<HloInstructionProto*> LookUpMutableInstruction(XlaOp op);
  StatusOr<HloInstructionProto*> LookUpMutableInstructionByHandle(
      int64_t handle);

  // Internal helper method that does the building for an arbitrary unary op.
  virtual XlaOp UnaryOp(HloOpcode unop, XlaOp operand);

  // Internal helper method that does the building for an arbitrary binary op.
  // broadcast_dimensions specifies which dimensions to use for broadcasting
  // when the operation is between tensors of different ranks. The direction is
  // only used if opcode is kCompare.
  XlaOp BinaryOp(HloOpcode binop, XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64_t> broadcast_dimensions,
                 std::optional<ComparisonDirection> direction = std::nullopt,
                 std::optional<Comparison::Type> type = std::nullopt);

  StatusOr<XlaOp> Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                          ComparisonDirection direction);

  // Internal helper method for binary op compare without broadcast dimensions.
  virtual StatusOr<XlaOp> Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                                  ComparisonDirection direction,
                                  Comparison::Type type);

  // Internal helper method that does the building for an arbitrary binary op
  // with same ranked operands that doesn't broadcast.
  virtual XlaOp BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape,
                                    XlaOp lhs, XlaOp rhs);

  // Internal helper method that does the building for an arbitrary ternary op.
  XlaOp TernaryOp(HloOpcode triop, XlaOp lhs, XlaOp rhs, XlaOp ehs);

  XlaOp RngOp(RandomDistribution distribution,
              absl::Span<const XlaOp> parameters, const Shape& shape);

  virtual StatusOr<XlaOp> RngOpInternal(RandomDistribution distribution,
                                        absl::Span<const XlaOp> parameters,
                                        const Shape& shape);

  virtual StatusOr<XlaOp> InDimBroadcast(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64_t> broadcast_dimensions);

  // Internal helper method that creates a sequence of instructions that
  // performs an explicit broadcast of the operand to the target shape.
  // All dimensions of the operand must either be equal to the corresponding
  // output shape dimension, or be exactly 1.  (Such dimensions are the
  // degenerate dimensions.)
  StatusOr<XlaOp> AddBroadcastSequence(const Shape& output_shape,
                                       XlaOp operand);

  // Internal helper method for creating a Reshape op with the already inferred
  // shape.
  virtual StatusOr<XlaOp> ReshapeInternal(const Shape& shape, XlaOp operand,
                                          int64_t inferred_dimension);

  // Returns the (inferred) result for the program shape using the given root.
  StatusOr<ProgramShape> GetProgramShape(int64_t root_id) const;

  // A visitor which checks whether an operation is a compile-time constant,
  // meaning that it doesn't depend on any parameters, or on any stateful
  // operation such as `RngNormal` or `Infeed`. The visitor walks the
  // computation starting at a given operation and sets is_constant to false iff
  // a parameter or stateful operation is encountered.
  void IsConstantVisitor(int64_t op_handle, int depth,
                         absl::flat_hash_set<int64_t>* visited,
                         bool* is_constant) const;

  // Checks bounds for convolution parameters.
  Status VerifyConvolution(
      const Shape& lhs_shape, const Shape& rhs_shape,
      const ConvolutionDimensionNumbers& dimension_numbers) const;

  int64_t GetNextId() { return ++next_id_; }

  // Populates the module with the input/output alias information stored within
  // the input_output_aliases vector.
  static Status PopulateInputOutputAliasAndBufferDonor(
      HloModuleProto* module, const ProgramShape& program_shape,
      const std::vector<InputOutputAlias>& input_output_aliases,
      const absl::flat_hash_set<HloBufferDonorConfig::BufferDonor>&
          buffer_donors);

  std::string name_;  // Name to use for the built computation.

  // The next sequential ID for every instruction/computation contained within
  // this computation.
  int64_t next_id_ = 0;

  // The first error encountered while building the computation.
  // This is OK until the first error is encountered.
  Status first_error_;

  // The saved stack trace from the point at which the first error occurred.
  tsl::SavedStackTrace first_error_backtrace_;

  // The instructions of this computation.
  // Use a deque so pointers into this are stable, for example the return
  // value of LookUpInstructionByHandle().
  std::deque<HloInstructionProto> instructions_;
  // A cache for the HloInstructionProto shapes, to avoid recreating Shape
  // objects from protos and to support the GetShapePtr() API.
  std::vector<std::unique_ptr<Shape>> instruction_shapes_;

  // Dynamic parameter configuration of this computation.
  DynamicParameterBinding dynamic_parameter_binding_;

  // Holds the input/output alias information populated by the SetUpAlias() API.
  std::vector<InputOutputAlias> input_output_aliases_;

  // Holds the buffer donor information populated by the AddBufferDonor() API.
  absl::flat_hash_set<HloBufferDonorConfig::BufferDonor> buffer_donors_;

  // A map from XlaOp::Handle to the index in the instructions_ vector where the
  // instruction is held.
  absl::flat_hash_map<int64_t, int64_t> handle_to_index_;

  // Track imported instructions by their computation id and the position in
  // their computation's instruction list.
  struct ImportedInstruction {
    int64_t computation_id;
    int64_t instruction_index;
  };

  absl::flat_hash_map<int64_t, ImportedInstruction> handle_to_imported_index_;

  // The embedded computations used by this computation. Each computation was
  // the entry computation of some XlaComputation, the key is the unique id of
  // that XlaComputation.
  std::map<int64_t, HloComputationProto> embedded_;

  // The unique parameter numbers.
  absl::flat_hash_set<int64_t> parameter_numbers_;

  // The metadata to attach to each op. This is structured as a "modal"-like
  // operation, in order to simplify client code (and not sprinkle this metadata
  // throughout the TensorFlow op kernel implementations).
  OpMetadata metadata_;

  // A temporary metadata that will only be applied to the next op created.
  std::optional<OpMetadata> one_shot_metadata_;

  // Sharding for this operator. This is structured as a "model"-like operation,
  // in order to simplify client code, similar to metadata_.
  std::optional<OpSharding> sharding_;

  // Mode bit that indicates whether to die when a first error is encountered.
  bool die_immediately_on_error_ = false;

  XlaBuilder* parent_builder_{nullptr};

  FrontendAttributes frontend_attributes_;

  friend XlaOp Parameter(XlaBuilder* builder, int64_t parameter_number,
                         const Shape& shape, const std::string& name,
                         const std::vector<bool>& replicated_at_leaf_buffers);
  friend XlaOp ConstantLiteral(XlaBuilder* builder,
                               const LiteralSlice& literal);

  friend XlaOp Broadcast(XlaOp operand,
                         absl::Span<const int64_t> broadcast_sizes);

  friend XlaOp BroadcastInDim(XlaOp operand,
                              absl::Span<const int64_t> out_dim_size,
                              absl::Span<const int64_t> broadcast_dimensions);

  friend XlaOp Copy(XlaOp operand);

  friend XlaOp Pad(XlaOp operand, XlaOp padding_value,
                   const PaddingConfig& padding_config);

  friend XlaOp PadInDim(XlaOp operand, XlaOp padding_value, int64_t dimno,
                        int64_t pad_lo, int64_t pad_hi);

  friend XlaOp Reshape(XlaOp operand, absl::Span<const int64_t> dimensions,
                       absl::Span<const int64_t> new_sizes);

  friend XlaOp Reshape(XlaOp operand, absl::Span<const int64_t> new_sizes);

  friend XlaOp Reshape(const Shape& shape, XlaOp operand);

  friend XlaOp DynamicReshape(XlaOp operand, absl::Span<const XlaOp> dim_sizes,
                              absl::Span<const int64_t> new_size_bounds,
                              const std::vector<bool>& dims_are_dynamic);

  friend XlaOp ReshapeWithInferredDimension(XlaOp operand,
                                            absl::Span<const int64_t> new_sizes,
                                            int64_t inferred_dimension);

  friend XlaOp Collapse(XlaOp operand, absl::Span<const int64_t> dimensions);

  friend XlaOp Slice(XlaOp operand, absl::Span<const int64_t> start_indices,
                     absl::Span<const int64_t> limit_indices,
                     absl::Span<const int64_t> strides);

  friend XlaOp SliceInDim(XlaOp operand, int64_t start_index,
                          int64_t limit_index, int64_t stride, int64_t dimno);

  friend XlaOp DynamicSlice(XlaOp operand,
                            absl::Span<const XlaOp> start_indices,
                            absl::Span<const int64_t> slice_sizes);

  friend XlaOp DynamicUpdateSlice(XlaOp operand, XlaOp update,
                                  absl::Span<const XlaOp> start_indices);

  friend XlaOp ConcatInDim(XlaBuilder* builder,
                           absl::Span<const XlaOp> operands, int64_t dimension);

  friend XlaOp Select(XlaOp pred, XlaOp on_true, XlaOp on_false);
  friend XlaOp Tuple(XlaBuilder* builder, absl::Span<const XlaOp> elements);
  friend XlaOp GetTupleElement(XlaOp tuple_data, int64_t index);
  friend XlaOp Compare(XlaOp lhs, XlaOp rhs,
                       absl::Span<const int64_t> broadcast_dimensions,
                       ComparisonDirection direction);
  friend XlaOp Compare(XlaOp lhs, XlaOp rhs,
                       absl::Span<const int64_t> broadcast_dimensions,
                       ComparisonDirection direction,
                       Comparison::Type compare_type);
  friend XlaOp Dot(XlaOp lhs, XlaOp rhs,
                   const PrecisionConfig* precision_config,
                   std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp DotGeneral(XlaOp lhs, XlaOp rhs,
                          const DotDimensionNumbers& dimension_number,
                          const PrecisionConfig* precision_config,
                          std::optional<PrimitiveType> preferred_element_type);
  virtual StatusOr<XlaOp> DotGeneralInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs,
      const DotDimensionNumbers& dimension_number,
      const PrecisionConfig* precision_config);
  friend XlaOp Conv(XlaOp lhs, XlaOp rhs,
                    absl::Span<const int64_t> window_strides, Padding padding,
                    int64_t feature_group_count, int64_t batch_group_count,
                    const PrecisionConfig* precision_config,
                    std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp ConvWithGeneralPadding(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config,
      std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp ConvWithGeneralDimensions(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config,
      std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp ConvGeneral(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config,
      std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp DynamicConvForward(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp DynamicConvKernelGrad(
      XlaOp activations, XlaOp gradients,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type);
  friend XlaOp DynamicConvInputGrad(
      XlaOp input_sizes, XlaOp lhs, XlaOp rhs,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config, PaddingType padding_type,
      std::optional<PrimitiveType> preferred_element_type);

  friend XlaOp ConvKernelGrad(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config,
      std::optional<PrimitiveType> preferred_element_type);

  friend XlaOp ConvGeneralDilated(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config,
      std::optional<PrimitiveType> preferred_element_type,
      std::optional<std::vector<bool>> window_reversal);

  friend XlaOp Fft(XlaOp operand, FftType fft_type,
                   absl::Span<const int64_t> fft_length);
  friend XlaOp TriangularSolve(XlaOp a, XlaOp b, bool left_side, bool lower,
                               bool unit_diagonal,
                               TriangularSolveOptions::Transpose transpose_a);
  friend XlaOp Cholesky(XlaOp a, bool lower);
  friend XlaOp Infeed(XlaBuilder* builder, const Shape& shape,
                      const std::string& config);
  friend void Outfeed(XlaOp operand, const Shape& shape_with_layout,
                      const std::string& outfeed_config);
  friend XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
                    absl::Span<const XlaOp> operands);
  friend XlaOp CustomCall(
      XlaBuilder* builder, const std::string& call_target_name,
      absl::Span<const XlaOp> operands, const Shape& shape,
      const std::string& opaque, bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, CustomCallSchedule schedule,
      CustomCallApiVersion api_version);
  friend XlaOp CustomCallWithComputation(
      XlaBuilder* builder, const std::string& call_target_name,
      absl::Span<const XlaOp> operands, const XlaComputation& computation,
      const Shape& shape, const std::string& opaque, bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, CustomCallSchedule schedule,
      CustomCallApiVersion api_version);
  friend XlaOp CustomCallWithLayout(
      XlaBuilder* builder, const std::string& call_target_name,
      absl::Span<const XlaOp> operands, const Shape& shape_with_layout,
      absl::Span<const Shape> operand_shapes_with_layout,
      const std::string& opaque, bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, CustomCallSchedule schedule,
      CustomCallApiVersion api_version);
  friend XlaOp CustomCallWithConvDnums(
      XlaBuilder* builder, const std::string& call_target_name,
      absl::Span<const XlaOp> operands, const Shape& shape,
      absl::Span<const Shape> operand_shapes_with_layout,
      const std::string& opaque, bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, Window window, ConvolutionDimensionNumbers dnums,
      CustomCallSchedule schedule, CustomCallApiVersion api_version);
  friend XlaOp OptimizationBarrier(XlaOp operand);
  friend XlaOp Complex(XlaOp real, XlaOp imag,
                       absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Conj(XlaOp operand);
  friend XlaOp Add(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Sub(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Mul(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Div(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Rem(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Max(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Min(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp And(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Or(XlaOp lhs, XlaOp rhs,
                  absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Xor(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Not(XlaOp operand);
  friend XlaOp PopulationCount(XlaOp operand);
  friend XlaOp ShiftLeft(XlaOp lhs, XlaOp rhs,
                         absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp ShiftRightArithmetic(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp ShiftRightLogical(
      XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Reduce(XlaOp operand, XlaOp init_value,
                      const XlaComputation& computation,
                      absl::Span<const int64_t> dimensions_to_reduce);
  friend XlaOp Reduce(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                      absl::Span<const XlaOp> init_values,
                      const XlaComputation& computation,
                      absl::Span<const int64_t> dimensions_to_reduce);
  friend XlaOp ReduceAll(XlaOp operand, XlaOp init_value,
                         const XlaComputation& computation);
  friend XlaOp ReduceWindow(XlaOp operand, XlaOp init_value,
                            const XlaComputation& computation,
                            absl::Span<const int64_t> window_dimensions,
                            absl::Span<const int64_t> window_strides,
                            Padding padding);
  friend XlaOp ReduceWindow(absl::Span<const XlaOp> operands,
                            absl::Span<const XlaOp> init_values,
                            const XlaComputation& computation,
                            absl::Span<const int64_t> window_dimensions,
                            absl::Span<const int64_t> window_strides,
                            Padding padding);
  friend XlaOp ReduceWindowWithGeneralPadding(
      XlaOp operand, XlaOp init_value, const XlaComputation& computation,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const int64_t> base_dilations,
      absl::Span<const int64_t> window_dilations,
      absl::Span<const std::pair<int64_t, int64_t>> padding);
  friend XlaOp ReduceWindowWithGeneralPadding(
      absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
      const XlaComputation& computation,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const int64_t> base_dilations,
      absl::Span<const int64_t> window_dilations,
      absl::Span<const std::pair<int64_t, int64_t>> padding);

  friend XlaOp CrossReplicaSum(XlaOp operand,
                               absl::Span<const ReplicaGroup> replica_groups);
  friend XlaOp AllGather(XlaOp operand, int64_t all_gather_dimension,
                         int64_t shard_count,
                         absl::Span<const ReplicaGroup> replica_groups,
                         const std::optional<ChannelHandle>& channel_id,
                         const std::optional<Layout>& layout,
                         std::optional<bool> use_global_device_ids);
  friend XlaOp AllReduce(XlaOp operand, const XlaComputation& computation,
                         absl::Span<const ReplicaGroup> replica_groups,
                         const std::optional<ChannelHandle>& channel_id,
                         const std::optional<Shape>& shape_with_layout,
                         std::optional<bool> use_global_device_ids);
  friend XlaOp AllReduceTuple(absl::Span<const XlaOp> operand,
                              const XlaComputation& computation,
                              absl::Span<const ReplicaGroup> replica_groups,
                              const std::optional<ChannelHandle>& channel_id,
                              const std::optional<Shape>& shape_with_layout,
                              std::optional<bool> use_global_device_ids);
  friend XlaOp ReduceScatter(XlaOp operand, const XlaComputation& computation,
                             int64_t scatter_dimension, int64_t shard_count,
                             absl::Span<const ReplicaGroup> replica_groups,
                             const std::optional<ChannelHandle>& channel_id,
                             const std::optional<Layout>& layout,
                             std::optional<bool> use_global_device_ids);

  friend XlaOp AllToAll(XlaOp operand, int64_t split_dimension,
                        int64_t concat_dimension, int64_t split_count,
                        absl::Span<const ReplicaGroup> replica_groups,
                        const std::optional<Layout>& layout,
                        const std::optional<ChannelHandle>& channel_id);
  friend XlaOp AllToAllTuple(absl::Span<const XlaOp> operands,
                             absl::Span<const ReplicaGroup> replica_groups,
                             const std::optional<Layout>& layout,
                             const std::optional<ChannelHandle>& channel_id);
  friend XlaOp AllToAllTuple(XlaOp operand, int64_t split_dimension,
                             int64_t concat_dimension, int64_t split_count,
                             absl::Span<const ReplicaGroup> replica_groups,
                             const std::optional<Layout>& layout,
                             const std::optional<ChannelHandle>& channel_id);
  friend XlaOp CollectivePermute(
      XlaOp operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<ChannelHandle>& channel_id);
  friend XlaOp ReplicaId(XlaBuilder* builder);
  friend XlaOp SelectAndScatter(XlaOp operand, const XlaComputation& select,
                                absl::Span<const int64_t> window_dimensions,
                                absl::Span<const int64_t> window_strides,
                                Padding padding, XlaOp source, XlaOp init_value,
                                const XlaComputation& scatter);
  friend XlaOp SelectAndScatterWithGeneralPadding(
      XlaOp operand, const XlaComputation& select,
      absl::Span<const int64_t> window_dimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding, XlaOp source,
      XlaOp init_value, const XlaComputation& scatter);
  friend XlaOp Abs(XlaOp operand);
  friend XlaOp Atan2(XlaOp y, XlaOp x,
                     absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp Exp(XlaOp operand);
  friend XlaOp Expm1(XlaOp operand);
  friend XlaOp Floor(XlaOp operand);
  friend XlaOp Ceil(XlaOp operand);
  friend XlaOp Round(XlaOp operand);
  friend XlaOp RoundNearestEven(XlaOp operand);
  friend XlaOp Log(XlaOp operand);
  friend XlaOp Log1p(XlaOp operand);
  friend XlaOp Logistic(XlaOp operand);
  friend XlaOp Sign(XlaOp operand);
  friend XlaOp Clz(XlaOp operand);
  friend XlaOp Cos(XlaOp operand);
  friend XlaOp Sin(XlaOp operand);
  friend XlaOp Tan(XlaOp operand);
  friend XlaOp Tanh(XlaOp operand);
  friend XlaOp Real(XlaOp operand);
  friend XlaOp Imag(XlaOp operand);
  friend XlaOp Sqrt(XlaOp operand);
  friend XlaOp Rsqrt(XlaOp operand);
  friend XlaOp Cbrt(XlaOp operand);
  friend XlaOp Pow(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions);
  friend XlaOp IsFinite(XlaOp operand);
  friend XlaOp Iota(XlaBuilder* builder, const Shape& shape,
                    int64_t iota_dimension);
  friend XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64_t size);
  friend XlaOp ConvertElementType(XlaOp operand,
                                  PrimitiveType new_element_type);
  friend XlaOp BitcastConvertType(XlaOp operand,
                                  PrimitiveType new_element_type);
  friend XlaOp StochasticConvertType(XlaOp operand, XlaOp random,
                                     PrimitiveType new_element_type);
  friend XlaOp Neg(XlaOp operand);
  friend XlaOp Transpose(XlaOp operand, absl::Span<const int64_t> permutation);
  friend XlaOp Rev(XlaOp operand, absl::Span<const int64_t> dimensions);
  friend XlaOp Sort(absl::Span<const XlaOp> operands,
                    const XlaComputation& comparator, int64_t dimension,
                    bool is_stable);
  friend XlaOp TopK(XlaOp operand, int64_t k, bool largest);
  friend XlaOp Clamp(XlaOp min, XlaOp operand, XlaOp max);
  friend XlaOp Map(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                   const XlaComputation& computation,
                   absl::Span<const int64_t> dimensions,
                   absl::Span<const XlaOp> static_operands);
  friend XlaOp RngNormal(XlaOp mu, XlaOp sigma, const Shape& shape);
  friend XlaOp RngUniform(XlaOp a, XlaOp b, const Shape& shape);
  friend XlaOp RngBitGenerator(RandomAlgorithm algorithm, XlaOp initial_state,
                               const Shape& shape);
  friend XlaOp While(const XlaComputation& condition,
                     const XlaComputation& body, XlaOp init);
  friend XlaOp Conditional(XlaOp predicate, XlaOp true_operand,
                           const XlaComputation& true_computation,
                           XlaOp false_operand,
                           const XlaComputation& false_computation);
  friend XlaOp Conditional(
      XlaOp branch_index,
      absl::Span<const XlaComputation* const> branch_computations,
      absl::Span<const XlaOp> branch_operands);
  friend XlaOp ConditionalImpl(
      XlaOp branch_index,
      absl::Span<const XlaComputation* const> branch_computations,
      absl::Span<const XlaOp> branch_operands);
  friend XlaOp ReducePrecision(XlaOp operand, int exponent_bits,
                               int mantissa_bits);
  friend XlaOp Gather(XlaOp input, XlaOp start_indices,
                      const GatherDimensionNumbers& dimension_numbers,
                      absl::Span<const int64_t> slice_sizes,
                      bool indices_are_sorted);
  friend XlaOp Scatter(XlaOp input, XlaOp scatter_indices, XlaOp updates,
                       const XlaComputation& update_computation,
                       const ScatterDimensionNumbers& dimension_numbers,
                       bool indices_are_sorted, bool unique_indices);
  friend XlaOp Scatter(absl::Span<const XlaOp> inputs, XlaOp scatter_indices,
                       absl::Span<const XlaOp> updates,
                       const XlaComputation& update_computation,
                       const ScatterDimensionNumbers& dimension_numbers,
                       bool indices_are_sorted, bool unique_indices);
  friend void Send(XlaOp operand, const ChannelHandle& handle);
  friend XlaOp Recv(XlaBuilder* builder, const Shape& shape,
                    const ChannelHandle& handle);
  friend XlaOp BatchNormTraining(XlaOp operand, XlaOp scale, XlaOp offset,
                                 float epsilon, int64_t feature_index);
  friend XlaOp BatchNormInference(XlaOp operand, XlaOp scale, XlaOp offset,
                                  XlaOp mean, XlaOp variance, float epsilon,
                                  int64_t feature_index);
  friend XlaOp BatchNormGrad(XlaOp operand, XlaOp scale, XlaOp batch_mean,
                             XlaOp batch_var, XlaOp grad_output, float epsilon,
                             int64_t feature_index);
  friend XlaOp SendWithToken(XlaOp operand, XlaOp token,
                             const ChannelHandle& handle);
  friend XlaOp RecvWithToken(XlaOp token, const Shape& shape,
                             const ChannelHandle& handle);
  friend XlaOp SendToHost(XlaOp operand, XlaOp token,
                          const Shape& shape_with_layout,
                          const ChannelHandle& handle);
  friend XlaOp RecvFromHost(XlaOp token, const Shape& shape,
                            const ChannelHandle& handle);
  friend XlaOp InfeedWithToken(XlaOp token, const Shape& shape,
                               const std::string& config);
  friend XlaOp OutfeedWithToken(XlaOp operand, XlaOp token,
                                const Shape& shape_with_layout,
                                const std::string& outfeed_config);
  friend XlaOp CreateToken(XlaBuilder* builder);
  friend XlaOp AfterAll(XlaBuilder* builder, absl::Span<const XlaOp> tokens);

  friend XlaOp GetDimensionSize(XlaOp operand, int64_t dimension);
  friend XlaOp SetDimensionSize(XlaOp operand, XlaOp val, int64_t dimension);
  friend XlaOp RemoveDynamicDimension(XlaOp operand, int64_t dimension);

 protected:
  // Returns OK status if the given op was built using this builder. Otherwise,
  // returns an error.
  Status CheckOpBuilder(XlaOp op) const;

 private:
  XlaOp AllGatherImpl(XlaOp operand, int64_t all_gather_dimension,
                      int64_t shard_count,
                      absl::Span<const ReplicaGroup> replica_groups,
                      const std::optional<ChannelHandle>& channel_id,
                      const std::optional<Layout>& layout,
                      std::optional<bool> use_global_device_ids, bool async);

  XlaOp AllReduceImpl(XlaOp operand, const XlaComputation& computation,
                      absl::Span<const ReplicaGroup> replica_groups,
                      const std::optional<ChannelHandle>& channel_id,
                      const std::optional<Shape>& layout,
                      std::optional<bool> use_global_device_ids, bool async);

  XlaOp CollectivePermuteImpl(
      XlaOp operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<ChannelHandle>& channel_id, bool async);

  XlaOp ConditionalImpl(
      XlaOp branch_index,
      absl::Span<const XlaComputation* const> branch_computations,
      absl::Span<const XlaOp> branch_operands);

  XlaOp AllToAllArray(
      XlaOp operand, int64_t split_dimension, int64_t concat_dimension,
      int64_t split_count, absl::Span<const ReplicaGroup> replica_groups,
      const std::optional<ChannelHandle>& channel_id = std::nullopt);

  // Creates an op with the given opcode and the output shape.
  virtual StatusOr<XlaOp> AddOpWithShape(HloOpcode opcode, const Shape& shape,
                                         absl::Span<const XlaOp> operands);

  // Here, InstructionType is either const HloInstructionProto* or non-const
  // HloInstructionProto*.
  template <typename InstructionType>
  StatusOr<InstructionType> LookUpInstructionByHandleInternal(
      int64_t handle) const {
    auto it = handle_to_index_.find(handle);
    if (it == handle_to_index_.end()) {
      // Try look for the instruction in the imported instructions.
      auto imported_it = handle_to_imported_index_.find(handle);
      if (imported_it != handle_to_imported_index_.end()) {
        ImportedInstruction imported = imported_it->second;
        return const_cast<InstructionType>(
            &embedded_.at(imported.computation_id)
                 .instructions(imported.instruction_index));
      }
      return InvalidArgument("No XlaOp with handle %d", handle);
    }
    return const_cast<InstructionType>(&instructions_.at(it->second));
  }

  // Here, InstructionType is either const HloInstructionProto* or non-const
  // HloInstructionProto*.
  //
  // TODO(hinsu): Return const pointer within StatusOr and use
  // absl::implicit_cast at callsites. This requires implicit_cast support in
  // xla::StatusOr similar to absl::StatusOr.
  template <typename InstructionType>
  StatusOr<InstructionType> LookUpInstructionInternal(XlaOp op) const {
    TF_RETURN_IF_ERROR(CheckOpBuilder(op));
    return LookUpInstructionByHandleInternal<InstructionType>(op.handle());
  }

  friend struct internal::XlaBuilderFriend;

  friend class ValueInference;
};

// RAII-style object: sets the current sharding assignment in builder on
// construction, and sets back to the previous assignment on destruction.
class XlaScopedShardingAssignment {
 public:
  XlaScopedShardingAssignment(xla::XlaBuilder* builder,
                              std::optional<OpSharding> sharding)
      : builder_(builder), prev_sharding_(builder->sharding()) {
    SetSharding(sharding);
  }

  XlaScopedShardingAssignment(const XlaScopedShardingAssignment&) = delete;
  XlaScopedShardingAssignment& operator=(const XlaScopedShardingAssignment&) =
      delete;

  ~XlaScopedShardingAssignment() { SetSharding(prev_sharding_); }

 private:
  void SetSharding(const std::optional<OpSharding>& sharding) {
    if (sharding.has_value()) {
      builder_->SetSharding(sharding.value());
    } else {
      builder_->ClearSharding();
    }
  }

  xla::XlaBuilder* const builder_;
  std::optional<OpSharding> prev_sharding_;
};

// RAII-style object: save the current builder's frontend attributes, and merge
// them with the new ones on construction.
// Restore the original attributes on destruction.
class XlaScopedFrontendAttributesAssignment {
 public:
  XlaScopedFrontendAttributesAssignment(xla::XlaBuilder* builder,
                                        FrontendAttributes attributes)
      : builder_(builder) {
    saved_ = builder_->SwapFrontendAttributes(attributes);
  }

  ~XlaScopedFrontendAttributesAssignment() {
    builder_->SetFrontendAttributes(saved_);
  }

 private:
  xla::XlaBuilder* const builder_;
  FrontendAttributes saved_;

  XlaScopedFrontendAttributesAssignment(
      const XlaScopedFrontendAttributesAssignment&) = delete;
  XlaScopedFrontendAttributesAssignment& operator=(
      const XlaScopedFrontendAttributesAssignment&) = delete;
};

// RAII-style object: sets the current op metadata in builder on construction,
// and sets back to the previous assignment on destruction.
class XlaScopedOpMetadataAssignment {
 public:
  XlaScopedOpMetadataAssignment(xla::XlaBuilder* builder, OpMetadata metadata)
      : builder_(builder) {
    saved_ = builder_->SwapOpMetadata(metadata);
  }

  ~XlaScopedOpMetadataAssignment() { builder_->SwapOpMetadata(saved_); }

 private:
  xla::XlaBuilder* const builder_;
  OpMetadata saved_;

  XlaScopedOpMetadataAssignment(const XlaScopedOpMetadataAssignment&) = delete;
  XlaScopedOpMetadataAssignment& operator=(
      const XlaScopedOpMetadataAssignment&) = delete;
};

// Free functions for building XlaOps. The intention is that these will
// become the public API for building XlaOps rather than calling methods on
// XlaBuilder directly.
//

// Enqueues a "retrieve parameter value" instruction for a parameter that was
// passed to the computation.
XlaOp Parameter(XlaBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name);

// Same as above, but with leaf buffer replication annotation.
XlaOp Parameter(XlaBuilder* builder, int64_t parameter_number,
                const Shape& shape, const std::string& name,
                const std::vector<bool>& replicated_at_leaf_buffers);

// Enqueues a constant with the value of the given literal onto the
// computation.
XlaOp ConstantLiteral(XlaBuilder* builder, const LiteralSlice& literal);

// Enqueues a constant onto the computation. Methods are templated on the
// native host type (NativeT) which corresponds to a specific XLA
// PrimitiveType as given in the following table:
//
//  Native Type   PrimitiveType
// -----------------------------
//   bool           PRED
//   int32_t        S32
//   int64_t        S64
//   uint32_t       U32
//   uint64_t       U64
//   float          F32
//   double         F64
//
// Note: not all primitive types defined in xla_data.proto have a
// corresponding native type yet.
template <typename NativeT>
XlaOp ConstantR0(XlaBuilder* builder, NativeT value);
template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder, absl::Span<const NativeT> values);
XlaOp ConstantR1(XlaBuilder* builder, const tsl::core::Bitmap& values);
template <typename NativeT>
XlaOp ConstantR2(XlaBuilder* builder,
                 std::initializer_list<std::initializer_list<NativeT>> values);
template <typename NativeT>
XlaOp ConstantFromArrayWithLayout(XlaBuilder* builder,
                                  const Array<NativeT>& values,
                                  const Layout& layout);
template <typename NativeT>
XlaOp ConstantFromArray(XlaBuilder* builder, const Array<NativeT>& values);
template <typename NativeT>
XlaOp ConstantR2FromArray2DWithLayout(XlaBuilder* builder,
                                      const Array2D<NativeT>& values,
                                      const Layout& layout);
template <typename NativeT>
XlaOp ConstantR2FromArray2D(XlaBuilder* builder,
                            const Array2D<NativeT>& values);
template <typename NativeT>
XlaOp ConstantR3FromArray3DWithLayout(XlaBuilder* builder,
                                      const Array3D<NativeT>& values,
                                      const Layout& layout);
template <typename NativeT>
XlaOp ConstantR3FromArray3D(XlaBuilder* builder,
                            const Array3D<NativeT>& values);
template <typename NativeT>
XlaOp ConstantR4FromArray4DWithLayout(XlaBuilder* builder,
                                      const Array4D<NativeT>& values,
                                      const Layout& layout);
template <typename NativeT>
XlaOp ConstantR4FromArray4D(XlaBuilder* builder,
                            const Array4D<NativeT>& values);

// Enqueues a rank one constant (XlaBuilder* builder, vector) onto the
// computation. The vector has size 'length' and every element has the value
// 'value'.
template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder, int64_t length, NativeT value);

// Adds dimensions to an array by duplicating the data in the array.
//
// The new dimensions are inserted on the left, i.e. if
// broadcast_sizes has values {a0, ..., aN} and the operand shape
// has dimensions {b0, ..., bM} then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
//
// The new dimensions index into copies of the operand, i.e.
//
//   output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
XlaOp Broadcast(XlaOp operand, absl::Span<const int64_t> broadcast_sizes);

// This op broadcasts the `operand` to an output with the given `shape`.
// `broadcast_dimensions` are the dimensions to be broadcasting into, i.e., the
// i'th dimension of the operand is mapped to the broadcast_dimensions[i]'th
// dimension of the output. This also requires that the i'th input dimension is
// either 1 or is the same as the output dimension it's broadcasting into.
//
// For example, say operand = {1, 2}, i.e., a 1D tensor in shape s32[2]; the
// output shape is s32[2,2]:
// - Specifying {1} as broadcast_dimension will generate output
//   {{1, 2},
//    {1, 2}}
// - On the other hand, specifying {0} as broadcast_dimension
//   will generate output
//   {{1 , 1},
//    {2 , 2}}
XlaOp BroadcastInDim(XlaOp operand, absl::Span<const int64_t> out_dim_size,
                     absl::Span<const int64_t> broadcast_dimensions);

// Copies the input operand to the output. This operation is for internal
// purpose and is only used by the compiler for optimization purposes or to
// ensure correctness. The XLA client should never have to generate this
// instruction.
//
// Copy has two potential use cases:
//
// * Create a copy of the operand with a new layout.
//
// * Create a copy of the operand in a separately allocated buffer. This is
//   necessary for some backends if the operand is a parameter or constant and
//   the operand is returned within a tuple. In this case, the lifetime of the
//   operand buffer must be the same as the lifetime of the output result.
//   However, the lifetimes of parameters and constants are managed separately
//   from the lifetime of the output result. Creating a separate copy of the
//   parameter or constant buffer resolves this issue.
XlaOp Copy(XlaOp operand);

// Enqueues a pad operation onto the computation that pads the given value on
// the edges as well as between the elements of the input. padding_config
// specifies the padding amount for each dimension.
XlaOp Pad(XlaOp operand, XlaOp padding_value,
          const PaddingConfig& padding_config);

// Enqueues a pad operation in a given dimension, taking all other
// dimensions as they are.
XlaOp PadInDim(XlaOp operand, XlaOp padding_value, int64_t dimno,
               int64_t pad_lo, int64_t pad_hi);

// Enqueues an operation onto the computation that flattens the operand based
// on the dimension order (major/slowest-varying to minor/fastest-varying)
// given, followed by reshaping it into the shape with the given dimension
// sizes (also major to minor). Conceptually, this is a limited form of
// "shape casting".
XlaOp Reshape(XlaOp operand, absl::Span<const int64_t> dimensions,
              absl::Span<const int64_t> new_sizes);

// Enqueues a dynamic reshape operation. The dynamic reshape takes additional
// XlaOps as sizes for the result dimension. The result dim i is a dynamic
// dimension dimension if dims_are_dynamic[i] is true.
XlaOp DynamicReshape(XlaOp operand, absl::Span<const XlaOp> dim_sizes,
                     absl::Span<const int64_t> new_size_bounds,
                     const std::vector<bool>& dims_are_dynamic);

// Enqueues an operation onto the computation that collapses the operand,
// from first to last dimension (C order), then reshapes it to the given
// dimension sizes. Conceptually, this is a limited form of "shape casting".
XlaOp Reshape(XlaOp operand, absl::Span<const int64_t> new_sizes);

// Enqueues a Reshape op that uses an explicit target shape.
XlaOp Reshape(const Shape& shape, XlaOp operand);

// `inferred_dimension` represents the output dimension that's inferred by
// upper-level framework by dividing the input element count by the known
// output element count. While an inferred_dimension can be static, if there
// is a dynamic dimension in the output, it must be the inferred dimension.
XlaOp ReshapeWithInferredDimension(XlaOp operand,
                                   absl::Span<const int64_t> new_sizes,
                                   int64_t inferred_dimension);

// Wrapper for Reshape.
// Enqueues an operation to collapse the provided dimensions; e.g. an
// operand with dimensions {x=256, y=2, z=2, p=32} can be collapsed to
// {x=1024, y=32} by collapsing dims {0, 1, 2}. Collapsing dimensions must
// be a consecutive, in-order subsequence of the operand dimensions.
//
// Note that collapsing a single dimension does nothing:
//
//    {256} collapsing {0} => {256}
//    {1} collapsing {0} => {1}
//
// Collapsing multiple dimensions produces a single result dimension:
//
//    {256, 2} collapsing {0,1} => {512}
//    {256, 2, 3} collapsing {0,1} => {512, 3}
//
// This could potentially cause data to be moved -- it provides a more
// structured form of reshaping than an arbitrary Reshape operation.
XlaOp Collapse(XlaOp operand, absl::Span<const int64_t> dimensions);

// Enqueues a slice operation onto the computation that slices the operand
// from the start indices to the limit indices; e.g.
//
//        x
//   [ 0 1 2 3 ]
// y [ 4 5 6 7 ] => slice(start={1, 1}, limit={2, 3}) => [ 5 6 ]
//   [ 8 9 a b ]
//
// Note that "limit" means up-to-but-not-including; i.e. [start, limit) in 1D
// range notation.
// The strides parameter determines the stride over the slice
XlaOp Slice(XlaOp operand, absl::Span<const int64_t> start_indices,
            absl::Span<const int64_t> limit_indices,
            absl::Span<const int64_t> strides);

// Enqueues a slice operation in a given dimension, taking all other
// dimensions as they are; e.g. if dimno is 1 from start_index 2 to
// limit_index 4 by 1, and the shape is f32[7,8,9], this call is short-hand
// for:
//
//  array[:, 2:4:1, :]
XlaOp SliceInDim(XlaOp operand, int64_t start_index, int64_t limit_index,
                 int64_t stride, int64_t dimno);

// Enqueues a slice operation onto the computation that slices the 'operand'
// from dynamic start indices which are passed in 'start_indices'.
// The size of the slice in each dimension is passed in 'slice_sizes',
// which specify the end point of exclusive slice intervals in each
// dimension [start, start + size).
// The shape of each element of 'start_indices' must be scalar, with the span
// size equal to the rank of the 'operand'. All elements of 'start_indices' must
// have the same shape.
// Slice index calculations are computed modulo input dimension sizes to
// prevent dynamic start indices from generating out-of-bound array accesses.
XlaOp DynamicSlice(XlaOp operand, absl::Span<const XlaOp> start_indices,
                   absl::Span<const int64_t> slice_sizes);

// Enqueues a dynamic update slice operation onto the computation, which
// updates a slice of 'operand' with 'update' at dynamic 'start_indices'.
// The shape of 'update' determines the shape of the slice of 'operand'
// which is updated.
// The indices specified in 'start_indices' specify the offset of the slice
// of 'operand' which is updated.
//
//               update = {10, 11} // calculated at runtime.
//   [1 2 3]     start  = {1, 1}   // calculated at runtime.  [1 2  3 ]
//   [4 5 6]  => DynamicUpdateslice(data, update, start)   => [4 10 11]
//   [7 8 9]                                                  [7 8  9 ]
//
// The shape of each element of 'start_indices' must be scalar, with the span
// size equal to the rank of the 'operand'. All elements of 'start_indices' must
// have the same shape.
// Slice index calculations are computed modulo update dimension sizes to
// prevent dynamic start indices from generating out-of-bound array accesses.
XlaOp DynamicUpdateSlice(XlaOp operand, XlaOp update,
                         absl::Span<const XlaOp> start_indices);

// Enqueues a concatenate instruction onto the computation. 'operands' must
// have >= 1 entry.
XlaOp ConcatInDim(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                  int64_t dimension);

// Enqueues a conditional-move-like select operation onto the computation;
// predicated on pred, selects between on_true and on_false.
XlaOp Select(XlaOp pred, XlaOp on_true, XlaOp on_false);

// Enqueues a tuple-creation instruction onto the computation.
XlaOp Tuple(XlaBuilder* builder, absl::Span<const XlaOp> elements);

// Enqueues a tuple-element-get instruction onto the computation.
XlaOp GetTupleElement(XlaOp tuple_data, int64_t index);

// Enqueues an equal-to comparison instruction onto the computation.
XlaOp Eq(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp EqTotalOrder(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a not-equal comparison instruction onto the computation.
XlaOp Ne(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp NeTotalOrder(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a greater-or-equal comparison instruction onto the computation.
XlaOp Ge(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp GeTotalOrder(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a greater-than comparison instruction onto the computation.
XlaOp Gt(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp GtTotalOrder(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a less-than comparison instruction onto the computation.
XlaOp Lt(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp LtTotalOrder(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a less-or-equal comparison instruction onto the computation.
XlaOp Le(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp LeTotalOrder(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a comparison instruction onto the computation (optionally without
// broadcast_dimensions for consistency with others).
XlaOp Compare(XlaOp lhs, XlaOp rhs,
              absl::Span<const int64_t> broadcast_dimensions,
              ComparisonDirection direction, Comparison::Type compare_type);
XlaOp Compare(XlaOp lhs, XlaOp rhs,
              absl::Span<const int64_t> broadcast_dimensions,
              ComparisonDirection direction);
XlaOp Compare(XlaOp lhs, XlaOp rhs, ComparisonDirection direction);

// Enqueues a dot instruction onto the computation.
XlaOp Dot(XlaOp lhs, XlaOp rhs,
          const PrecisionConfig* precision_config = nullptr,
          std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues a general dot instruction onto the computation.
XlaOp DotGeneral(
    XlaOp lhs, XlaOp rhs, const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig* precision_config = nullptr,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues a convolution instruction onto the computation, which uses the
// default convolution dimension numbers.
XlaOp Conv(XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
           Padding padding, int64_t feature_group_count = 1,
           int64_t batch_group_count = 1,
           const PrecisionConfig* precision_config = nullptr,
           std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues a convolution instruction onto the computation, with the caller
// provided padding configuration in the format returned by MakePadding().
XlaOp ConvWithGeneralPadding(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    int64_t feature_group_count = 1, int64_t batch_group_count = 1,
    const PrecisionConfig* precision_config = nullptr,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues a convolution instruction onto the computation, with the caller
// provided dimension numbers configuration.
XlaOp ConvWithGeneralDimensions(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count = 1, int64_t batch_group_count = 1,
    const PrecisionConfig* precision_config = nullptr,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues a convolution instruction onto the computation, with the caller
// provided padding configuration as well as the dimension numbers.
XlaOp ConvGeneral(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count = 1, int64_t batch_group_count = 1,
    const PrecisionConfig* precision_config = nullptr,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues a convolution instruction onto the computation, with the caller
// provided padding configuration, dilation factors and dimension numbers.
XlaOp ConvGeneralDilated(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count = 1, int64_t batch_group_count = 1,
    const PrecisionConfig* precision_config = nullptr,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt,
    std::optional<std::vector<bool>> window_reversal = std::nullopt);

XlaOp DynamicConvForward(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

XlaOp DynamicConvInputGrad(
    XlaOp input_sizes, XlaOp lhs, XlaOp rhs,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

XlaOp DynamicConvKernelGrad(
    XlaOp activations, XlaOp gradients,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64_t feature_group_count, int64_t batch_group_count,
    const PrecisionConfig* precision_config, PaddingType padding_type,
    std::optional<PrimitiveType> preferred_element_type = std::nullopt);

// Enqueues an FFT instruction onto the computation, of the given type and
// with the given FFT length.
XlaOp Fft(XlaOp operand, FftType fft_type,
          absl::Span<const int64_t> fft_length);

// Solves systems of linear equations with lower or upper triangular coefficient
// matrices by forward- or back-substitution. Broadcasting along leading
// dimensions, this routine solves for x in one of the matrix systems
//   `op(a) * x = b`,  or `x * op(a) = b`,
// for the variable `x` given `a` and `b`, where `op(a)` is either
//   `op(a) = a`,  or `op(a) = transpose(a)`,  or `op(a) = conj(transpose(a))`.
//
// * `a` is a tensor of shape `[..., M, M]` whose innermost 2 dimensions form
//   square matrices. If `lower` is true (false), then the strictly upper
//   (lower) triangular part of each innermost matrix in `a` is assumed to be
//   zero and is not accessed.
// * `b` is a tensor of shape `[..., M, K]` if `left_side` is true, otherwise a
//   tensor of shape `[..., K, M]`.
// * `left_side` is a boolean, indicating whether to solve a system of the form
//   op(a) * x = b (true) or x * op(a) = b (false).
// * `lower` is a boolean, indicating whether the argument `a` is
//   lower-triangular (true) or upper-triangular (false).
// * If `unit_diagonal` is true, the diagonal elements of `a` are assumed to be
//   1 and not accessed.
// * `transpose_a` indicates which function `op` we use to transform the tensor
//   `a`: the identity function, transpose(a), or conjugate(transpose(a))
XlaOp TriangularSolve(XlaOp a, XlaOp b, bool left_side, bool lower,
                      bool unit_diagonal,
                      TriangularSolveOptions::Transpose transpose_a);

// Computes the Cholesky decompositions of a batch of symmetric (Hermitian)
// positive definite matrices.
// `a` must be a (batched) square matrix; i.e., it must have rank >= 2 with the
// two minor dimensions equal.
// If `lower` is true, the data from the lower triangle is used; if false, the
// upper triangle is used. The input data in the other triangle of the input
// does not affect the output. Returns the output in the same lower/upper
// triangle. The data returned in the other output triangle is arbitrary and
// implementation-defined.
//
// If `a` is not Hermitian positive definite, returns an array full of NaNs.
XlaOp Cholesky(XlaOp a, bool lower);

// Enqueues an infeed instruction onto the computation, which writes data of
// the given shape to the infeed buffer of the device.
XlaOp Infeed(XlaBuilder* builder, const Shape& shape,
             const std::string& config = "");

// Variant of Infeed which takes a token-shaped operand and produces a
// two-element tuple containing the data value and a token-shaped value.
// Tokens are used for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp InfeedWithToken(XlaOp token, const Shape& shape,
                      const std::string& config = "");

// Enqueues an outfeed instruction onto the computation. This instruction
// generates outgoing data transfers for the given data.
//
// shape_with_layout communicates the laid out shape that we want to outfeed
// -- if !ShapeUtil::Compatible(GetShape(operand), shape_with_layout) an error
// will occur.
void Outfeed(XlaOp operand, const Shape& shape_with_layout,
             const std::string& outfeed_config);

// Variant of Outfeed which takes a token-shaped operand and produces a
// token-shaped value. Tokens are used for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp OutfeedWithToken(XlaOp operand, XlaOp token,
                       const Shape& shape_with_layout,
                       const std::string& outfeed_config);

// Enqueues a call instruction onto the computation.
XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
           absl::Span<const XlaOp> operands);

// Enqueues a custom call instruction onto the computation. A custom call
// invokes code external to XLA. The |operands| are passed to the external code,
// and the external code is expected to produce a result of the given
// |shape|. The exact mechanism is backend-specific. For example, in the CPU
// backend, a call instruction is emitted which targets a symbol with the name
// |call_target_name|.  |call_target_name| and |opaque| can arbitrary strings,
// but |call_target_name| should be short as it may be used in labels. |opaque|
// can encode arbitrarily large amounts of information. |has_side_effect|
// specifies whether the instruction can have side effects.
// |output_operand_aliasing| specifies a list of output/operand buffer pairs
// that alias each other, where the output buffer is represented as a
// ShapeIndex, and the operand buffer is represented as the operand index and
// the ShapeIndex.
XlaOp CustomCall(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const Shape& shape,
    const std::string& opaque = "", bool has_side_effect = false,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing = {},
    const Literal* literal = nullptr,
    CustomCallSchedule schedule = CustomCallSchedule::SCHEDULE_NONE,
    CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

// Overload which constructs a custom call that applies an Xla computation.
XlaOp CustomCallWithComputation(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const XlaComputation& computation,
    const Shape& shape, const std::string& opaque = "",
    bool has_side_effect = false,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing = {},
    const Literal* literal = nullptr,
    CustomCallSchedule schedule = CustomCallSchedule::SCHEDULE_NONE,
    CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

// Overload which constructs a custom call with fixed layouts. The operands will
// have the layouts specified by |operand_shapes_with_layout| when provided to
// external code, and the external code is expected to produce a result with the
// layout specified by |shape_with_layout|. All shapes in |shape_with_layout|
// and |operand_shapes_with_layout| must have layouts.
XlaOp CustomCallWithLayout(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const Shape& shape_with_layout,
    absl::Span<const Shape> operand_shapes_with_layout,
    const std::string& opaque = "", bool has_side_effect = false,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing = {},
    const Literal* literal = nullptr,
    CustomCallSchedule schedule = CustomCallSchedule::SCHEDULE_NONE,
    CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

// Overload which annotates a custom call with the given Window and
// ConvolutionDimensionNumbers.  Useful for custom-calls which represent
// convolutions.
//
// This sets the layout of its operands if operand_shapes_with_layout is
// nonempty, and it sets the layout of its result if `shape` has a layout.
XlaOp CustomCallWithConvDnums(
    XlaBuilder* builder, const std::string& call_target_name,
    absl::Span<const XlaOp> operands, const Shape& shape,
    absl::Span<const Shape> operand_shapes_with_layout,
    const std::string& opaque, bool has_side_effect,
    absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_operand_aliasing,
    const Literal* literal, Window window, ConvolutionDimensionNumbers dnums,
    CustomCallSchedule schedule = CustomCallSchedule::SCHEDULE_NONE,
    CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

// Enqueues an optimization barrier onto the computation.
XlaOp OptimizationBarrier(XlaOp operand);

// The following methods enqueue element-wise binary arithmetic operations
// onto the computation. The shapes of the operands have to match unless one
// of the operands is a scalar, or an explicit broadcast dimension is given
// (see g3doc for more details).

// Enqueues a complex compose instruction onto the computation.
XlaOp Complex(XlaOp real, XlaOp imag,
              absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a complex conjugate instruction onto the computation.
XlaOp Conj(XlaOp operand);

// Enqueues an add instruction onto the computation.
XlaOp Add(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a subtract instruction onto the computation.
XlaOp Sub(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a multiply instruction onto the computation.
XlaOp Mul(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a divide instruction onto the computation.
XlaOp Div(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a remainder instruction onto the computation.
XlaOp Rem(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a max instruction onto the computation.
XlaOp Max(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues a min instruction onto the computation.
XlaOp Min(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Element-wise logical operators
XlaOp And(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Overload to call And with 3 or more operands.  We need the following somewhat
// convoluted overload set to disambiguate with the overload that takes the
// `broadcast_dimensions` optional param.
inline XlaOp And(XlaOp op1, XlaOp op2, XlaOp op3) {
  return And(op1, And(op2, op3));
}
template <typename... XlaOpTs>
XlaOp And(XlaOp op1, XlaOp op2, XlaOp op3, const XlaOpTs&... operands) {
  return And(op1, And(op2, And(op3, operands...)));
}

XlaOp Or(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64_t> broadcast_dimensions = {});

// Overload to call Or with 3 or more operands.  As with `And`, we need the
// following complicated overload set to handle the default arg in the `Or`
// overload above.
inline XlaOp Or(XlaOp op1, XlaOp op2, XlaOp op3) {
  return Or(op1, Or(op2, op3));
}
template <typename... XlaOpTs>
XlaOp Or(XlaOp op1, XlaOp op2, XlaOp op3, const XlaOpTs&... operands) {
  return Or(op1, Or(op2, Or(op3, operands...)));
}

XlaOp Xor(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

XlaOp Not(XlaOp operand);

XlaOp PopulationCount(XlaOp operand);

XlaOp ShiftLeft(XlaOp lhs, XlaOp rhs,
                absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp ShiftRightArithmetic(XlaOp lhs, XlaOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions = {});
XlaOp ShiftRightLogical(XlaOp lhs, XlaOp rhs,
                        absl::Span<const int64_t> broadcast_dimensions = {});
// Reduces an array among the provided dimensions, given "computation" as a
// reduction operator.
XlaOp Reduce(XlaOp operand, XlaOp init_value, const XlaComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce);

// Reduces several arrays simultaneously among the provided dimensions, given
// "computation" as a reduction operator.
XlaOp Reduce(XlaBuilder* builder, absl::Span<const XlaOp> operands,
             absl::Span<const XlaOp> init_values,
             const XlaComputation& computation,
             absl::Span<const int64_t> dimensions_to_reduce);

// Convenience wrapper around the above that reduces all the dimensions in the
// operand shape.
XlaOp ReduceAll(XlaOp operand, XlaOp init_value,
                const XlaComputation& computation);

// Enqueues a windowed reduce instruction onto the computation.
XlaOp ReduceWindow(XlaOp operand, XlaOp init_value,
                   const XlaComputation& computation,
                   absl::Span<const int64_t> window_dimensions,
                   absl::Span<const int64_t> window_strides, Padding padding);

XlaOp ReduceWindow(absl::Span<const XlaOp> operands,
                   absl::Span<const XlaOp> init_values,
                   const XlaComputation& computation,
                   absl::Span<const int64_t> window_dimensions,
                   absl::Span<const int64_t> window_strides, Padding padding);

// As ReduceWindow(), but the padding is given in the format
// returned by MakePadding().
XlaOp ReduceWindowWithGeneralPadding(
    XlaOp operand, XlaOp init_value, const XlaComputation& computation,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> base_dilations,
    absl::Span<const int64_t> window_dilations,
    absl::Span<const std::pair<int64_t, int64_t>> padding);
XlaOp ReduceWindowWithGeneralPadding(
    absl::Span<const XlaOp> operands, absl::Span<const XlaOp> init_values,
    const XlaComputation& computation,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const int64_t> base_dilations,
    absl::Span<const int64_t> window_dilations,
    absl::Span<const std::pair<int64_t, int64_t>> padding);

// Returns the sum of the operand value within each subgroup of replicas. All
// replicas supply one input to the sum and all replicas receive the resulting
// sum for each subgroup.
XlaOp CrossReplicaSum(XlaOp operand,
                      absl::Span<const ReplicaGroup> replica_groups = {});

XlaOp AllGather(XlaOp operand, int64_t all_gather_dimension,
                int64_t shard_count,
                absl::Span<const ReplicaGroup> replica_groups = {},
                const std::optional<ChannelHandle>& channel_id = std::nullopt,
                const std::optional<Layout>& layout = std::nullopt,
                std::optional<bool> use_global_device_ids = std::nullopt);

// Enqueues an operation that do an AllReduce of the operand cross cores. Here
// AllReduce means doing a reduction on the input operand cross cores and then
// broadcasting the reduction result to those cores. The reduction function is
// defined by `computation`, which should be a commutative computation on
// scalars, e.g., add, min, or max. The way that AllReduce is applied is
// configured by:
//
// - `replica_groups`: each ReplicaGroup contains a list of replica id. If
// empty, all replicas belong to one group. Allreduce will be applied within
// subgroups. For example, we have 4 replicas, then replica_groups={{0,2},{1,3}}
// means, replica 0 and 2 are in subgroup 0, replica 1 and 3 are in subgroup 1.
//
// - `channel_id`: for Allreduce nodes from different modules, if they have the
// same channel_id, they will be 'AllReduce'd. If empty, AllReduce will not be
// applied cross modules.
//
// - `shape_with_layout`: forces the layout of the AllReduce to the given
// layout. This is used to guarantee the same layout for a group of AllReduce
// ops compiled separately.
XlaOp AllReduce(XlaOp operand, const XlaComputation& computation,
                absl::Span<const ReplicaGroup> replica_groups = {},
                const std::optional<ChannelHandle>& channel_id = std::nullopt,
                const std::optional<Shape>& shape_with_layout = std::nullopt,
                std::optional<bool> use_global_device_ids = std::nullopt);

XlaOp AllReduceTuple(
    absl::Span<const XlaOp> operand, const XlaComputation& computation,
    absl::Span<const ReplicaGroup> replica_groups = {},
    const std::optional<ChannelHandle>& channel_id = std::nullopt,
    const std::optional<Shape>& shape_with_layout = std::nullopt,
    std::optional<bool> use_global_device_ids = std::nullopt);

XlaOp ReduceScatter(
    XlaOp operand, const XlaComputation& computation, int64_t scatter_dimension,
    int64_t shard_count, absl::Span<const ReplicaGroup> replica_groups = {},
    const std::optional<ChannelHandle>& channel_id = std::nullopt,
    const std::optional<Layout>& layout = std::nullopt,
    std::optional<bool> use_global_device_ids = std::nullopt);

// Enqueues an operation that do an Alltoall of the operand cross cores.
// An optional `layout` can be specified to force the layout of the instruction.
// This is used to guarantee the same layout for a group of AllToAll ops
// compiled separately.
XlaOp AllToAll(XlaOp operand, int64_t split_dimension, int64_t concat_dimension,
               int64_t split_count,
               absl::Span<const ReplicaGroup> replica_groups = {},
               const std::optional<Layout>& layout = std::nullopt,
               const std::optional<ChannelHandle>& channel_id = std::nullopt);

XlaOp AllToAllTuple(
    absl::Span<const XlaOp> operand,
    absl::Span<const ReplicaGroup> replica_groups = {},
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<ChannelHandle>& channel_id = std::nullopt);

XlaOp AllToAllTuple(
    XlaOp operand, int64_t split_dimension, int64_t concat_dimension,
    int64_t split_count, absl::Span<const ReplicaGroup> replica_groups = {},
    const std::optional<Layout>& layout = std::nullopt,
    const std::optional<ChannelHandle>& channel_id = std::nullopt);

// Enqueues an collective operation that sends and receives data cross replicas.
//
// - `source_target_pair`: a list of (source_replica_id, target_replica_id)
// pairs. For each pair, the operand is sent from source replica to target
// replica. Note that, 1) any two pairs should not have the same target replica
// id, and they should not have the same source replica id; 2) if a replica id
// is not a target in any pair, then the output on that replica is a tensor
// consists of 0(s) with the same shape as the input.
XlaOp CollectivePermute(
    XlaOp operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<ChannelHandle>& channel_id = std::nullopt);

// Enqueues an operation that returns the replica ID.
XlaOp ReplicaId(XlaBuilder* builder);

// Enqueues an operation that scatters the `source` array to the selected
// indices of each window.
XlaOp SelectAndScatter(XlaOp operand, const XlaComputation& select,
                       absl::Span<const int64_t> window_dimensions,
                       absl::Span<const int64_t> window_strides,
                       Padding padding, XlaOp source, XlaOp init_value,
                       const XlaComputation& scatter);

// As SelectAndScatter(), but the padding is given in the format
// returned by MakePadding().
XlaOp SelectAndScatterWithGeneralPadding(
    XlaOp operand, const XlaComputation& select,
    absl::Span<const int64_t> window_dimensions,
    absl::Span<const int64_t> window_strides,
    absl::Span<const std::pair<int64_t, int64_t>> padding, XlaOp source,
    XlaOp init_value, const XlaComputation& scatter);

// Enqueues an abs instruction onto the computation.
XlaOp Abs(XlaOp operand);

// Enqueues a atan2 instruction onto the computation.
XlaOp Atan2(XlaOp y, XlaOp x,
            absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues an exp instruction onto the computation.
XlaOp Exp(XlaOp operand);

// Enqueues an expm1 instruction onto the computation.
XlaOp Expm1(XlaOp operand);

// Enqueues a floor instruction onto the computation.
XlaOp Floor(XlaOp operand);

// Enqueues a ceil instruction onto the computation.
XlaOp Ceil(XlaOp operand);

// Enqueues a round instruction onto the computation,
// with half-way cases rounding away from zero.
XlaOp Round(XlaOp operand);

// Enqueues a round instruction onto the computation, rounding to nearest even
XlaOp RoundNearestEven(XlaOp operand);

// Enqueues an log instruction (natural logarithm) onto the computation.
XlaOp Log(XlaOp operand);

// Enqueues an log1p instruction (log(x+1)) onto the computation.
XlaOp Log1p(XlaOp operand);

// Enqueues a logistic instruction onto the computation.
XlaOp Logistic(XlaOp operand);

// Enqueues a sign instruction onto the computation.
XlaOp Sign(XlaOp operand);

// Enqueues a count leading zeros instruction onto the computation.
XlaOp Clz(XlaOp operand);

// Enqueues a cosine instruction onto the computation.
XlaOp Cos(XlaOp operand);

// Enqueues a sine instruction onto the computation.
XlaOp Sin(XlaOp operand);

// Enqueues a tan instruction onto the computation.
XlaOp Tan(XlaOp operand);

// Enqueues a tanh instruction onto the computation.
XlaOp Tanh(XlaOp operand);

// Enqueues a real-part instruction onto the computation.
XlaOp Real(XlaOp operand);

// Enqueues an imaginary-part instruction onto the computation.
XlaOp Imag(XlaOp operand);

// Enqueues a sqrt computation onto the computation.
XlaOp Sqrt(XlaOp operand);

// Enqueues a cbrt computation onto the computation.
XlaOp Cbrt(XlaOp operand);

// Enqueues a rsqrt computation onto the computation.
XlaOp Rsqrt(XlaOp operand);

// Enqueues a lhs^rhs computation onto the computation.
XlaOp Pow(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Enqueues an operator that tests if the operand's values are finite, i.e., not
// +/-Inf or NaN.  Returns an array of booleans with the same shape where
// entries are true iff the corresponding entry was not infinite or NaN.
//
// Defined only for real-valued (i.e. not complex) floating-point types; raises
// an error for other types.
//
// See also IsInf, IsPosInf, IsNegInf, and IsNan in lib/math.h.
XlaOp IsFinite(XlaOp operand);

// Enqueues an iota operation onto the computation.
XlaOp Iota(XlaBuilder* builder, const Shape& shape, int64_t iota_dimension);

// Enqueues a rank-1 iota operation onto the computation.
XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64_t size);

// Enqueues a convert instruction onto the computation that changes the
// element type of the operand array to primitive_type.
XlaOp ConvertElementType(XlaOp operand, PrimitiveType new_element_type);

// Enqueues a no-op instruction onto the computation that changes
// the element type of the operand array to primitive_type. The
// bit-widths of the source and destination element types must be
// identical.
XlaOp BitcastConvertType(XlaOp operand, PrimitiveType new_element_type);

// Enqueues a stochastic convert instruction onto the computation that changes
// the element type of the operand array with stochastic rounding to
// primitive_type.
XlaOp StochasticConvertType(XlaOp operand, XlaOp random,
                            PrimitiveType new_element_type);

// Enqueues a negate instruction onto the computation.
XlaOp Neg(XlaOp operand);

// Enqueues a transpose instruction onto the computation.
XlaOp Transpose(XlaOp operand, absl::Span<const int64_t> permutation);

// Enqueues a reverse instruction onto the computation. The order of the
// elements in the given dimensions is reversed (i.e., the element at index i
// is moved to index dimension_size - 1 - i).
XlaOp Rev(XlaOp operand, absl::Span<const int64_t> dimensions);

// Enqueues a sort instruction onto the computation, using 'comparator' for
// comparisons. 'comparator' needs to define a strict weak order. 'is_stable'
// determines whether the stable sorting should be used.
// If only one operand is provided:
// * If the operand is a rank-1 tensor (an array), the result is a sorted array.
//   The resulting sorting order has the property that for all index positions
//   i, j with i < j, either
//   comparator(value[i], value[j]) = comparator(value[j], value[i]) = false or
//   comparator(value[i], value[j]) = true.
// * If the operand has higher rank, the operand is sorted along the provided
//   dimension. For example, for a rank-2 tensor (a matrix), a dimension value
//   of 0 will independently sort every column, and a dimension value of 1 will
//   independently sort each row. If no dimension number is provided, then the
//   last dimension is chosen by default. For the dimension which is sorted, the
//   same sorting order applies as in the rank-1 case.
//
// If more than one operand is provided:
// * All operands must be tensors with the same dimensions. The element types of
//   the tensors may be different.
// * The result is a tuple that consists of the operands in sorted order (along
//   the provided dimension, as above). The same permutation as implied by the
//   comparison computation is applied to all operand tensors. When comparing
//   two index positions, 'comparator' is called with 2 * n scalar parameters,
//   where parameter 2 * i and 2 * i + 1 correspond to the value of operand i at
//   two index positions.
// Default comparator computations can be found in lib/comparators.h
XlaOp Sort(absl::Span<const XlaOp> operands, const XlaComputation& comparator,
           int64_t dimension = -1, bool is_stable = false);

// Enqueues a topk instruction onto the computation. TopK returns the largest
// 'k' values and their indices along the last dimension of the 'operand' if
// `lagest=true` or the smallest `k` values if `largest=false`.
//
// * If the operand is a rank-1 tensor (an array), the result is a tuple that
//   consists of:
//   * a sorted array with the top 'k' elements.
//   * an array containing the indices of the k elements.
//   For example, if the input is [0.1, 0.3, 0.2] and k == 2, the output tuple
//   is ([0.3, 0.2], [1, 2]).
// * If the operand has higher rank, the result is a tuple that consists of:
//   * a tensor equivalent to one produced by sorting the operand along the last
//     dimension and slicing that dimension to only the top 'k' values. The last
//     dimension is sorted as in the rank-1 case.
//   * a tensor containing the indices of the top 'k' values along the last
//     dimension.
//   For example, if the input is [0.1, 0.3, 0.2][0.5, 0.4, 0.6] and k == 1, the
//   output tuple is ([0.3][0.6], [1][2]).
XlaOp TopK(XlaOp operand, int64_t k, bool largest);

// Enqueues a clamp instruction onto the computation.
XlaOp Clamp(XlaOp min, XlaOp operand, XlaOp max);

// Enqueues a map instruction onto the computation.
XlaOp Map(XlaBuilder* builder, absl::Span<const XlaOp> operands,
          const XlaComputation& computation,
          absl::Span<const int64_t> dimensions,
          absl::Span<const XlaOp> static_operands = {});

// Enqueues a N(mu, sigma) random number generation instruction onto the
// computation.
XlaOp RngNormal(XlaOp mu, XlaOp sigma, const Shape& shape);

// Enqueues a U(a, b) random number generation instruction onto the
// computation. Returns values in the semi-open interval [a, b).
XlaOp RngUniform(XlaOp a, XlaOp b, const Shape& shape);

// Enqueues a B(initial_state) random bit generation instruction onto the
// computation. Returns the new key and random bits with the specified shape.
XlaOp RngBitGenerator(RandomAlgorithm algorithm, XlaOp initial_state,
                      const Shape& shape);

// Enqueues a while node onto the computation.
XlaOp While(const XlaComputation& condition, const XlaComputation& body,
            XlaOp init);

// Enqueues a conditional node onto the computation.
XlaOp Conditional(XlaOp predicate, XlaOp true_operand,
                  const XlaComputation& true_computation, XlaOp false_operand,
                  const XlaComputation& false_computation);

// Enqueues either a predicated (if/else) or indexed (switch/case/default)
// conditional node onto the computation. N >= 1 branch_computations and
// branch_operands are matched by index. branch_index selects the branch that
// will be executed. Out of range branch_index uses the N-1'th
// branch_computation as default.
XlaOp Conditional(XlaOp branch_index,
                  absl::Span<const XlaComputation* const> branch_computations,
                  absl::Span<const XlaOp> branch_operands);

// Enqueues a ReducePrecision node onto the computation.
XlaOp ReducePrecision(XlaOp operand, int exponent_bits, int mantissa_bits);

// Enqueues a Gather node onto the computation.
XlaOp Gather(XlaOp input, XlaOp start_indices,
             const GatherDimensionNumbers& dimension_numbers,
             absl::Span<const int64_t> slice_sizes,
             bool indices_are_sorted = false);

// Enqueues a Scatter node onto the computation.
XlaOp Scatter(XlaOp input, XlaOp scatter_indices, XlaOp updates,
              const XlaComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted = false, bool unique_indices = false);
XlaOp Scatter(absl::Span<const XlaOp> inputs, XlaOp scatter_indices,
              absl::Span<const XlaOp> updates,
              const XlaComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers,
              bool indices_are_sorted = false, bool unique_indices = false);

// Enqueues a Send node onto the computation for device-to-device
// communication. This operation sends the given operand to
// a Recv instruction in a different computation that shares the same channel
// handle.
void Send(XlaOp operand, const ChannelHandle& handle);

// Variant of Send which takes a token-shaped operand and produces a
// token-shaped value.  Tokens are used for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp SendWithToken(XlaOp operand, XlaOp token, const ChannelHandle& handle);

// Enqueues a Recv node onto the computation for device-to-device
// communication. The data comes from a Send instruction in a different
// computation that shares the same channel handle and its shape must be the
// same as the given shape.
XlaOp Recv(XlaBuilder* builder, const Shape& shape,
           const ChannelHandle& handle);

// Variant of Recv which takes a token-shaped operand and produces a two-element
// tuple containing the data value and a token-shaped value. Tokens are used
// for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp RecvWithToken(XlaOp token, const Shape& shape,
                    const ChannelHandle& handle);

// Enqueues a Send node which transfers data from the device to the host. The
// 'shape_with_layout' argument defines the layout of the data transferred; its
// shape must be compatible with the shape of the operand. The operand must be
// array-shaped.
// TODO(b/111544877): Support tuple shapes.
XlaOp SendToHost(XlaOp operand, XlaOp token, const Shape& shape_with_layout,
                 const ChannelHandle& handle);

// Enqueues a Recv node which transfers data from the host to the device. The
// given shape must contain a layout and must be an array.
// TODO(b/111544877): Support tuple shapes.
XlaOp RecvFromHost(XlaOp token, const Shape& shape,
                   const ChannelHandle& handle);

// Enqueues an operation (AfterAll) with no operands that produces a
// token-shaped value.  Tokens are used for ordering side-effecting operations.
// This is a separate method from AfterAll to facility the removal of
// operand-less AfterAll instructions.
// TODO(b/110532604): Remove this function when all tokens are derived from a
// single token generated or passed into the entry computation.
XlaOp CreateToken(XlaBuilder* builder);

// Enqueues an AfterAll instruction which produces a token-shaped value and
// takes a variadic number of token-shaped operands. The number of operands must
// be greater than zero. Used for joining tokens.
XlaOp AfterAll(XlaBuilder* builder, absl::Span<const XlaOp> tokens);

// Normalizes operand across spatial and batch dimensions for each feature.
//
// Returns a tuple (normalized, batch_mean, batch_var) where `normalized`
// is the normalized result and batch_mean and batch_var are the mean and
// variance, respectively, across batch for the operand.
XlaOp BatchNormTraining(XlaOp operand, XlaOp scale, XlaOp offset, float epsilon,
                        int64_t feature_index);

// Normalizes operand across spatial and batch dimensions for each feature.
//
// `BatchNormInference` is equivalent to calling `BatchNormTraining` without
// computing `mean` and `variance` for each batch inside the operation. It
// uses the input `mean` and `variance` instead as estimated values. The
// purpose of this op is to reduce latency in inference, hence the name
// `BatchNormInference`.
//
// The output has the same shape as `operand`, and contains the normalized
// values for each batch.
XlaOp BatchNormInference(XlaOp operand, XlaOp scale, XlaOp offset, XlaOp mean,
                         XlaOp variance, float epsilon, int64_t feature_index);

// Calculates the gradients of a batch norm op.
//
// The inputs `batch_mean` and `batch_var` represent the mean and variance
// across the batch.
//
// Returns a tuple of three elements:
//   - grad_operand: Gradient with respect to input `operand`
//   - grad_offset: Gradient with respect to input `offset`
//   - grad_scale: Gradient with respect to input `scale`
XlaOp BatchNormGrad(XlaOp operand, XlaOp scale, XlaOp batch_mean,
                    XlaOp batch_var, XlaOp grad_output, float epsilon,
                    int64_t feature_index);

// Returns the size of the given dimension of the operand. The operand must be
// array shaped.
XlaOp GetDimensionSize(XlaOp operand, int64_t dimension);

// Sets the size of the given dimension of the operand. The operand must be
// array shaped.  The result will have the same shape as the operand, but the
// given dimension will be dynamic (if not already).
XlaOp SetDimensionSize(XlaOp operand, XlaOp val, int64_t dimension);

// Returns the same op but with dynamic dimension removed.
XlaOp RemoveDynamicDimension(XlaOp operand, int64_t dimension);

// Implementation details below this point.
//

// Free function template implementations.

template <typename NativeT>
XlaOp ConstantR0(XlaBuilder* builder, NativeT value) {
  return ConstantLiteral(builder, LiteralUtil::CreateR0<NativeT>(value));
}

template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder, absl::Span<const NativeT> values) {
  BorrowingLiteral literal(
      reinterpret_cast<const char*>(values.begin()),
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64_t>(values.size())}));
  return ConstantLiteral(builder, literal);
}

template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder, int64_t length, NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {length}));
  literal.PopulateWithValue(value);
  return ConstantLiteral(builder, literal);
}

inline XlaOp ConstantR1(XlaBuilder* builder, const tsl::core::Bitmap& values) {
  return ConstantLiteral(builder, LiteralUtil::CreateR1(values));
}

template <typename NativeT>
XlaOp ConstantR2(XlaBuilder* builder,
                 std::initializer_list<std::initializer_list<NativeT>> values) {
  return ConstantLiteral(builder, LiteralUtil::CreateR2<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantFromArrayWithLayout(XlaBuilder* builder,
                                  const Array<NativeT>& values,
                                  const Layout& layout) {
  return ConstantLiteral(
      builder, LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp ConstantFromArray(XlaBuilder* builder, const Array<NativeT>& values) {
  return ConstantLiteral(builder,
                         LiteralUtil::CreateFromArray<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantR2FromArray2DWithLayout(XlaBuilder* builder,
                                      const Array2D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantLiteral(
      builder, LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp ConstantR2FromArray2D(XlaBuilder* builder,
                            const Array2D<NativeT>& values) {
  return ConstantLiteral(builder,
                         LiteralUtil::CreateR2FromArray2D<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantR3FromArray3DWithLayout(XlaBuilder* builder,
                                      const Array3D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantLiteral(
      builder,
      LiteralUtil::CreateR3FromArray3DWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp ConstantR3FromArray3D(XlaBuilder* builder,
                            const Array3D<NativeT>& values) {
  return ConstantFromArray(builder, values);
}

template <typename NativeT>
XlaOp ConstantR4FromArray4DWithLayout(XlaBuilder* builder,
                                      const Array4D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantFromArrayWithLayout(builder, values, layout);
}

template <typename NativeT>
XlaOp ConstantR4FromArray4D(XlaBuilder* builder,
                            const Array4D<NativeT>& values) {
  return ConstantFromArray(builder, values);
}

// Switches from automatic SPMD partitioning to manual partitioning. Converts a
// full-shaped tensor (to be automatically partitioned by SPMD partitioner) to a
// shard-shaped tensor to be consumed by manually partitioned ops.
StatusOr<xla::XlaOp> ConvertSpmdFullToShardShape(
    xla::XlaBuilder* builder, xla::XlaOp input, int single_dim,
    const xla::OpSharding& manual_sharding,
    absl::Span<const int64_t> unspecified_dims);

// Switches from manual partitioning to automatic SPMD partitioning. Converts a
// shard-shaped tensor (manually partitioned in SPMD-style) to a full-shaped
// tensor to be partitioned automatically by the SPMD partitioner.
StatusOr<xla::XlaOp> ConvertSpmdShardToFullShape(
    xla::XlaBuilder* builder, xla::XlaOp input, const xla::Shape& output_shape,
    int single_dim, const xla::OpSharding& manual_sharding,
    absl::Span<const int64_t> unspecified_dims);

}  // namespace xla

#endif  // XLA_CLIENT_XLA_BUILDER_H_
