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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_BUILDER_H_

#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class XlaBuilder;

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

  bool IsIdenticalTo(const XlaOp& rhs) const {
    return handle_ == rhs.handle_ && builder_ == rhs.builder_;
  }

  friend std::ostream& operator<<(std::ostream& out, const XlaOp& op) {
    out << op.handle();
    return out;
  }

 private:
  explicit XlaOp(XlaBuilder* builder) : handle_(-1), builder_(builder) {}
  XlaOp(int64 handle, XlaBuilder* builder)
      : handle_(handle), builder_(builder) {}

  int64 handle() const { return handle_; }

  friend class XlaBuilder;

  // < 0 means "invalid handle".
  int64 handle_;

  // Not owned. Non-null for any handle returned by XlaBuilder, even if the
  // handle is invalid.
  XlaBuilder* builder_;
};

// Arithmetic operator overloads for the XlaOp type.
XlaOp operator-(const XlaOp& x);
XlaOp operator+(const XlaOp& x, const XlaOp& y);
XlaOp operator-(const XlaOp& x, const XlaOp& y);
XlaOp operator*(const XlaOp& x, const XlaOp& y);
XlaOp operator/(const XlaOp& x, const XlaOp& y);
XlaOp operator%(const XlaOp& x, const XlaOp& y);

// Bitwise operator overloads for the XlaOp type.
XlaOp operator~(const XlaOp& x);
XlaOp operator&(const XlaOp& x, const XlaOp& y);
XlaOp operator|(const XlaOp& x, const XlaOp& y);
XlaOp operator^(const XlaOp& x, const XlaOp& y);
XlaOp operator<<(const XlaOp& x, const XlaOp& y);
// Performs a right arithmetic shift if 'x' is a signed type, otherwise performs
// a right logical shift.
XlaOp operator>>(const XlaOp& x, const XlaOp& y);

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
  XlaBuilder(const string& computation_name);

  XlaBuilder(const XlaBuilder&) = delete;
  XlaBuilder& operator=(const XlaBuilder&) = delete;

  ~XlaBuilder();

  // Returns the computation name.
  const string& name() const { return name_; }

  // Sets OpMetadata that will be added to all instructions until cleared.
  //
  // OpMetadata is often applied to a series of XLA HLO instructions. As a
  // result, OpMetadata is set on the Computation Builder. All subsequent
  // instructions generated via this Computation Builder will have the same
  // OpMetadata attached until a call to ClearOpMetadata.
  void SetOpMetadata(const OpMetadata& metadata) { metadata_ = metadata; }

  // Clears the HloMetadata state.
  void ClearOpMetadata() { metadata_.Clear(); }

  // Sets an OpSharding that will be attached to all instructions until cleared.
  void SetSharding(const OpSharding& sharding) { sharding_ = sharding; }

  // Clears the sharding. Ops will be sharded according to the default placement
  // policy.
  void ClearSharding() { sharding_ = tensorflow::gtl::nullopt; }

  // Returns the OpSharding that will be attached to all instructions.
  const tensorflow::gtl::optional<OpSharding>& sharding() const {
    return sharding_;
  }

  // Sets the builder to a mode where it will die immediately when an error is
  // encountered, rather than producing it in a deferred fashion when Build() is
  // called (which is the default).
  void set_die_immediately_on_error(bool enabled) {
    die_immediately_on_error_ = enabled;
  }

  // Default dimension numbers used for a 2D convolution.
  static constexpr int64 kConvBatchDimension = 0;
  static constexpr int64 kConvFeatureDimension = 1;
  static constexpr int64 kConvFirstSpatialDimension = 2;
  static constexpr int64 kConvSecondSpatialDimension = 3;
  static constexpr int64 kConvKernelOutputDimension = 0;
  static constexpr int64 kConvKernelInputDimension = 1;
  static constexpr int64 kConvKernelFirstSpatialDimension = 2;
  static constexpr int64 kConvKernelSecondSpatialDimension = 3;

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
  std::unique_ptr<XlaBuilder> CreateSubBuilder(const string& computation_name);

  // Builds the computation with the requested operations, or returns a non-ok
  // status. Note that all ops that have been enqueued will be moved to the
  // computation being returned.
  StatusOr<XlaComputation> Build();

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
  StatusOr<XlaComputation> BuildConstantSubGraph(const XlaOp& root_op) const;

  // Returns the first error that was encountered while building the
  // computation. When an error is encountered, by default we return a vacuous
  // XlaOp and inform the user of the error that occurred while
  // building the computation when they make a final call to Build().
  //
  // See also set_die_immediately_on_error().
  Status first_error() const { return first_error_; }

  // Returns the shape of the given op.
  StatusOr<Shape> GetShape(const XlaOp& op) const;

  // Returns the (inferred) result for the current computation's shape.
  StatusOr<ProgramShape> GetProgramShape() const;

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
  XlaOp ReportErrorOrReturn(const std::function<StatusOr<XlaOp>()>& op_creator);

  // Returns true if 'operand' is a compile-time constant. A compile-time
  // constant does not depend on any parameters, or on stateful operators such
  // as `RngNormal` or `Infeed`.
  //
  // This tests whether a computation is a compile-time constant without
  // evaluating the computation.
  StatusOr<bool> IsConstant(const XlaOp& operand) const;

 private:
  // Enqueues a "retrieve parameter value" instruction for a parameter that was
  // passed to the computation.
  XlaOp Parameter(int64 parameter_number, const Shape& shape,
                  const string& name);

  // Enqueues a constant with the value of the given literal onto the
  // computation.
  XlaOp ConstantLiteral(const LiteralSlice& literal);

  // Enqueues a constant onto the computation. Methods are templated on the
  // native host type (NativeT) which corresponds to a specific XLA
  // PrimitiveType as given in the following table:
  //
  //  Native Type   PrimitiveType
  // -----------------------------
  //   bool           PRED
  //   int32          S32
  //   int64          S64
  //   uint32         U32
  //   uint64         U64
  //   float          F32
  //   double         F64
  //
  // Note: not all primitive types defined in xla_data.proto have a
  // corresponding native type yet.
  template <typename NativeT>
  XlaOp ConstantR0(NativeT value);
  template <typename NativeT>
  XlaOp ConstantR1(tensorflow::gtl::ArraySlice<NativeT> values);
  XlaOp ConstantR1(const tensorflow::core::Bitmap& values);
  template <typename NativeT>
  XlaOp ConstantR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  XlaOp ConstantFromArrayWithLayout(const Array<NativeT>& values,
                                    const Layout& layout);
  template <typename NativeT>
  XlaOp ConstantFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  XlaOp ConstantR2FromArray2DWithLayout(const Array2D<NativeT>& values,
                                        const Layout& layout);
  template <typename NativeT>
  XlaOp ConstantR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  XlaOp ConstantR3FromArray3DWithLayout(const Array3D<NativeT>& values,
                                        const Layout& layout);
  template <typename NativeT>
  XlaOp ConstantR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  XlaOp ConstantR4FromArray4DWithLayout(const Array4D<NativeT>& values,
                                        const Layout& layout);
  template <typename NativeT>
  XlaOp ConstantR4FromArray4D(const Array4D<NativeT>& values);

  // Enqueues a rank one constant (vector) onto the computation. The vector has
  // size 'length' and every element has the value 'value'.
  template <typename NativeT>
  XlaOp ConstantR1(int64 length, NativeT value);

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
  XlaOp Broadcast(const XlaOp& operand,
                  tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

  // Performs in-dimension-style broadcast.
  //
  // Operand specifies the input to be broadcast. "shape" is expected output
  // shape. "broadcast_dimensions" are the dimensions to be broadcasting into.
  // Dimension numbers in broadcast_dimensions map to individual dimensions
  // of the operand, and specify what dimension of the output shape they
  // should be broadcast.
  // e.g.
  // Say operand = [1, 2], i.e., a 1D tensor with 2 elements.
  // and dimension of shape is [2,2].
  // Specifying {1} as brodcast_dimension will generate output
  // [1 , 2]
  // [1 , 2]
  // On the other hand, specifying {0} as broadcast_dimension
  // will generate output
  // [1 , 1]
  // [2 , 2]
  XlaOp BroadcastInDim(
      const XlaOp& operand, const Shape& shape,
      const tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  // Enqueues a pad operation onto the computation that pads the given value on
  // the edges as well as between the elements of the input. padding_config
  // specifies the padding amount for each dimension.
  XlaOp Pad(const XlaOp& operand, const XlaOp& padding_value,
            const PaddingConfig& padding_config);

  // Enqueues an operation onto the computation that flattens the operand based
  // on the dimension order (major/slowest-varying to minor/fastest-varying)
  // given, followed by reshaping it into the shape with the given dimension
  // sizes (also major to minor). Conceptually, this is a limited form of
  // "shape casting".
  XlaOp Reshape(const XlaOp& operand,
                tensorflow::gtl::ArraySlice<int64> dimensions,
                tensorflow::gtl::ArraySlice<int64> new_sizes);

  // Enqueues an operation onto the computation that collapses the operand, from
  // first to last dimension (C order), then reshapes it to the given dimension
  // sizes. Conceptually, this is a limited form of "shape casting".
  XlaOp Reshape(const XlaOp& operand,
                tensorflow::gtl::ArraySlice<int64> new_sizes);

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
  XlaOp Collapse(const XlaOp& operand,
                 tensorflow::gtl::ArraySlice<int64> dimensions);

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
  XlaOp Slice(const XlaOp& operand,
              tensorflow::gtl::ArraySlice<int64> start_indices,
              tensorflow::gtl::ArraySlice<int64> limit_indices,
              tensorflow::gtl::ArraySlice<int64> strides);

  // Enqueues a slice operation in a given dimension, taking all other
  // dimensions as they are; e.g. if dimno is 1 from start_index 2 to
  // limit_index 4 by 1, and the shape is f32[7,8,9], this call is short-hand
  // for:
  //
  //  array[:, 2:4:1, :]
  XlaOp SliceInDim(const XlaOp& operand, int64 start_index, int64 limit_index,
                   int64 stride, int64 dimno);

  // Enqueues a slice operation onto the computation that slices the 'operand'
  // from dynamic start indices which are passed in 'start_indices'.
  // The size of the slice in each dimension is passed in 'slice_sizes',
  // which specify the end point of exclusive slice intervals in each
  // dimension [start, start + size).
  // The shape of 'start_indices' must be rank == 1, with dimension size
  // equal to the rank of the 'operand'.
  // Slice index calculations are computed modulo input dimension sizes to
  // prevent dynamic start indices from generating out-of-bound array accesses.
  XlaOp DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                     tensorflow::gtl::ArraySlice<int64> slice_sizes);

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
  // The shape of 'start_indices' must be rank == 1, with dimension size
  // equal to the rank of the 'operand'.
  // Slice index calculations are computed modulo update dimension sizes to
  // prevent dynamic start indices from generating out-of-bound array accesses.
  XlaOp DynamicUpdateSlice(const XlaOp& operand, const XlaOp& update,
                           const XlaOp& start_indices);

  // Enqueues a concatenate instruction onto the computation. 'operands' must
  // have >= 1 entry.
  XlaOp ConcatInDim(tensorflow::gtl::ArraySlice<XlaOp> operands,
                    int64 dimension);

  // Enqueue a tracing operation onto the computation; the computation will emit
  // a logging message with the operand.
  void Trace(const string& tag, const XlaOp& operand);

  // Enqueues a conditional-move-like select operation onto the computation;
  // predicated on pred, selects between on_true and on_false.
  XlaOp Select(const XlaOp& pred, const XlaOp& on_true, const XlaOp& on_false);

  // Enqueues a tuple-creation instruction onto the computation.
  XlaOp Tuple(tensorflow::gtl::ArraySlice<XlaOp> elements);

  // Enqueues a tuple-element-get instruction onto the computation.
  XlaOp GetTupleElement(const XlaOp& tuple_data, int64 index);

  // Enqueues an equal-to comparison instruction onto the computation.
  XlaOp Eq(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a not-equal comparison instruction onto the computation.
  XlaOp Ne(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a greater-or-equal comparison instruction onto the computation.
  XlaOp Ge(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a greater-than comparison instruction onto the computation.
  XlaOp Gt(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a less-than comparison instruction onto the computation.
  XlaOp Lt(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a less-or-equal comparison instruction onto the computation.
  XlaOp Le(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a dot instruction onto the computation.
  XlaOp Dot(const XlaOp& lhs, const XlaOp& rhs);

  // Enqueues a general dot instruction onto the computation.
  XlaOp DotGeneral(const XlaOp& lhs, const XlaOp& rhs,
                   const DotDimensionNumbers& dimension_numbers);

  // Enqueues a convolution instruction onto the computation, which uses the
  // default convolution dimension numbers.
  XlaOp Conv(const XlaOp& lhs, const XlaOp& rhs,
             tensorflow::gtl::ArraySlice<int64> window_strides,
             Padding padding);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration in the format returned by MakePadding().
  XlaOp ConvWithGeneralPadding(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided dimension numbers configuration.
  XlaOp ConvWithGeneralDimensions(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration as well as the dimension numbers.
  XlaOp ConvGeneral(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration, dilation factors and dimension numbers.
  XlaOp ConvGeneralDilated(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      tensorflow::gtl::ArraySlice<int64> lhs_dilation,
      tensorflow::gtl::ArraySlice<int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues an FFT instruction onto the computation, of the given type and
  // with the given FFT length.
  XlaOp Fft(const XlaOp& operand, FftType fft_type,
            tensorflow::gtl::ArraySlice<int64> fft_length);

  // Enqueues an infeed instruction onto the computation, which writes data of
  // the given shape to the infeed buffer of the device.
  XlaOp Infeed(const Shape& shape, const string& config = "");
  XlaOp InfeedWithToken(const XlaOp& token, const Shape& shape,
                        const string& config = "");

  // Enqueues an outfeed instruction onto the computation. This instruction
  // generates outgoing data transfers for the given data.
  //
  // shape_with_layout communicates the laid out shape that we want to outfeed
  // -- if !ShapeUtil::Compatible(GetShape(operand), shape_with_layout) an error
  // will occur.
  void Outfeed(const XlaOp& operand, const Shape& shape_with_layout,
               const string& outfeed_config);
  XlaOp OutfeedWithToken(const XlaOp& operand, const XlaOp& token,
                         const Shape& shape_with_layout,
                         const string& outfeed_config);

  // Enqueues a call instruction onto the computation.
  XlaOp Call(const XlaComputation& computation,
             tensorflow::gtl::ArraySlice<XlaOp> operands);

  // Enqueues a custom call instruction onto the computation.
  // During code generation, a call instruction is emitted which targets a
  // symbol with the name |call_target_name|.  The |operands| are passed to the
  // call instruction.  |shape| is the resultant shape.
  XlaOp CustomCall(const string& call_target_name,
                   tensorflow::gtl::ArraySlice<XlaOp> operands,
                   const Shape& shape);

  // Enqueues a pseudo-op to represent host-side computation data-dependencies.
  // During code generation, host send and receive operations will be generated
  // to transfer |operands| to the host and a single result of |shape| back to
  // the device.  Host send/recv operations are emitted using |channel_name|.
  // Dataflow dependencies and the |cost_estimate_ns| field may be used in HLO
  // instruction scheduling.
  XlaOp HostCompute(tensorflow::gtl::ArraySlice<XlaOp> operands,
                    const string& channel_name, int64 cost_estimate_ns,
                    const Shape& shape);

  // The following methods enqueue element-wise binary arithmetic operations
  // onto the computation. The shapes of the operands have to match unless one
  // of the operands is a scalar, or an explicit broadcast dimension is given
  // (see g3doc for more details).

  // Enqueues a complex compose instruction onto the computation.
  XlaOp Complex(const XlaOp& real, const XlaOp& imag,
                tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a complex conjugate instruction onto the computation.
  XlaOp Conj(const XlaOp& operand);

  // Enqueues an add instruction onto the computation.
  XlaOp Add(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a subtract instruction onto the computation.
  XlaOp Sub(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a multiply instruction onto the computation.
  XlaOp Mul(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a divide instruction onto the computation.
  XlaOp Div(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a remainder instruction onto the computation.
  XlaOp Rem(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a max instruction onto the computation.
  XlaOp Max(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a min instruction onto the computation.
  XlaOp Min(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Element-wise logical operators
  XlaOp And(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  XlaOp Or(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  XlaOp Xor(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  XlaOp Not(const XlaOp& operand);

  XlaOp ShiftLeft(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});
  XlaOp ShiftRightArithmetic(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});
  XlaOp ShiftRightLogical(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Reduces an array among the provided dimensions, given "computation" as a
  // reduction operator.
  XlaOp Reduce(const XlaOp& operand, const XlaOp& init_value,
               const XlaComputation& computation,
               tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce);

  // Convenience wrapper around the above that reduces all the dimensions in the
  // operand shape.
  XlaOp ReduceAll(const XlaOp& operand, const XlaOp& init_value,
                  const XlaComputation& computation);

  // Enqueues a windowed reduce instruction onto the computation.
  XlaOp ReduceWindow(const XlaOp& operand, const XlaOp& init_value,
                     const XlaComputation& computation,
                     tensorflow::gtl::ArraySlice<int64> window_dimensions,
                     tensorflow::gtl::ArraySlice<int64> window_strides,
                     Padding padding);

  // As ReduceWindow(), but the padding is given in the format
  // returned by MakePadding().
  XlaOp ReduceWindowWithGeneralPadding(
      const XlaOp& operand, const XlaOp& init_value,
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

  // Returns the sum of the operand value within each subgroup of replicas. All
  // replicas supply one input to the sum and all replicas receive the resulting
  // sum for each subgroup.
  XlaOp CrossReplicaSum(
      const XlaOp& operand,
      tensorflow::gtl::ArraySlice<int64> replica_group_ids = {});

  // Enqueues an operation that do an AllReduce of the operand cross cores. Here
  // AllReduce means doing a reduction on the input operand cross cores and then
  // broadcasting the reduction result to those cores. The reduction function is
  // defined by `computation`, which should be a commutative computation on
  // scalars, e.g., add, min, or max. The way that AllReduce is applied is
  // configured by:
  //
  // - `replica_group_ids`: maps replica ids to subgroup ids. If empty, all
  // replicas belong to one group. Allreduce will be applied within subgroups.
  // For example, we have 4 replicas, then replica_group_ids={0,1,0,1} means,
  // replica 0 and 2 are in subgroup 0, replica 1 and 3 are in subgroup 1.
  //
  // - `channel_id`: for Allreduce nodes from different models, if they have the
  // same channel_id, they will be 'Allreduce'd. If empty, Allreduce will not be
  // applied cross models.
  //
  // TODO(b/79737069): Rename this to AllReduce when it's ready to use.
  XlaOp CrossReplicaSum(
      const XlaOp& operand, const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<int64> replica_group_ids = {},
      const tensorflow::gtl::optional<ChannelHandle>& channel_id =
          tensorflow::gtl::nullopt);

  // Enqueues an operation that scatters the `source` array to the selected
  // indices of each window.
  XlaOp SelectAndScatter(const XlaOp& operand, const XlaComputation& select,
                         tensorflow::gtl::ArraySlice<int64> window_dimensions,
                         tensorflow::gtl::ArraySlice<int64> window_strides,
                         Padding padding, const XlaOp& source,
                         const XlaOp& init_value,
                         const XlaComputation& scatter);

  // As SelectAndScatter(), but the padding is given in the format
  // returned by MakePadding().
  XlaOp SelectAndScatterWithGeneralPadding(
      const XlaOp& operand, const XlaComputation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      const XlaOp& source, const XlaOp& init_value,
      const XlaComputation& scatter);

  // Enqueues an abs instruction onto the computation.
  XlaOp Abs(const XlaOp& operand);

  // Enqueues a atan2 instruction onto the computation.
  XlaOp Atan2(const XlaOp& y, const XlaOp& x,
              tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues an exp instruction onto the computation.
  XlaOp Exp(const XlaOp& operand);

  // Enqueues an expm1 instruction onto the computation.
  XlaOp Expm1(const XlaOp& operand);

  // Enqueues a floor instruction onto the computation.
  XlaOp Floor(const XlaOp& operand);

  // Enqueues a ceil instruction onto the computation.
  XlaOp Ceil(const XlaOp& operand);

  // Enqueues a round instruction onto the computation, rounding to nearest even
  // with half-way cases rounding away from zero.
  XlaOp Round(const XlaOp& operand);

  // Enqueues an log instruction (natural logarithm) onto the computation.
  XlaOp Log(const XlaOp& operand);

  // Enqueues an log1p instruction (log(x+1)) onto the computation.
  XlaOp Log1p(const XlaOp& operand);

  // Enqueues a sign instruction onto the computation.
  XlaOp Sign(const XlaOp& operand);

  // Enqueues a count leading zeros instruction onto the computation.
  XlaOp Clz(const XlaOp& operand);

  // Enqueues a cosine instruction onto the computation.
  XlaOp Cos(const XlaOp& operand);

  // Enqueues a sine instruction onto the computation.
  XlaOp Sin(const XlaOp& operand);

  // Enqueues a tanh instruction onto the computation.
  XlaOp Tanh(const XlaOp& operand);

  // Enqueues a real-part instruction onto the computation.
  XlaOp Real(const XlaOp& operand);

  // Enqueues an imaginary-part instruction onto the computation.
  XlaOp Imag(const XlaOp& operand);

  // Enqueues a lhs^rhs computation onto the computation.
  XlaOp Pow(const XlaOp& lhs, const XlaOp& rhs,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues an operator that tests if the operand's values are finite, i.e.,
  // not Inf or NaN. Defined only for floating-point types. Returns an array of
  // booleans with the same shape where entries are true iff the corresponding
  // entry was NaN.
  XlaOp IsFinite(const XlaOp& operand);

  // Enqueues a convert instruction onto the computation that changes the
  // element type of the operand array to primitive_type.
  XlaOp ConvertElementType(const XlaOp& operand,
                           PrimitiveType new_element_type);

  // Enqueues a no-op instruction onto the computation that changes
  // the element type of the operand array to primitive_type. The
  // bit-widths of the source and destination element types must be
  // identical.
  XlaOp BitcastConvertType(const XlaOp& operand,
                           PrimitiveType new_element_type);

  // Enqueues a negate instruction onto the computation.
  XlaOp Neg(const XlaOp& operand);

  // Enqueues a transpose instruction onto the computation.
  XlaOp Transpose(const XlaOp& operand,
                  tensorflow::gtl::ArraySlice<int64> permutation);

  // Enqueues a reverse instruction onto the computation. The order of the
  // elements in the given dimensions is reversed (i.e., the element at index i
  // is moved to index dimension_size - 1 - i).
  XlaOp Rev(const XlaOp& operand,
            tensorflow::gtl::ArraySlice<int64> dimensions);

  // Enqueues a sort (as increasing order) instruction onto the computation.
  // If only keys are provided:
  // * If the keys are an rank-1 tensor (an array), the result is a sorted array
  // of keys, in ascending order.
  // * If the keys have higher rank, the keys are sorted along the provided
  // dimension. For example, for a rank-2 tensor (a matrix) of keys, a dimension
  // value of 0 will indepenently sort every column, and a dimension value of 1
  // will independently sort each row. If no dimension number is provided, then
  // the last dimension is chosen by default.
  //
  // If both keys and values are provided:
  // * The keys and the values must tensors with the same dimensions. The
  // element types of the tensors may be different.
  // * The result is a tuple that consists of a sorted tensor of keys (along the
  // provided dimension, as above) as the first element, and a tensor with their
  // corresponding values as the second element.
  XlaOp Sort(XlaOp keys,
             tensorflow::gtl::optional<XlaOp> values = tensorflow::gtl::nullopt,
             int64 dimension = -1);

  // Enqueues a clamp instruction onto the computation.
  XlaOp Clamp(const XlaOp& min, const XlaOp& operand, const XlaOp& max);

  // Enqueues a map instruction onto the computation.
  XlaOp Map(tensorflow::gtl::ArraySlice<XlaOp> operands,
            const XlaComputation& computation,
            tensorflow::gtl::ArraySlice<int64> dimensions,
            tensorflow::gtl::ArraySlice<XlaOp> static_operands = {});

  // Enqueues a N(mu, sigma) random number generation instruction onto the
  // computation.
  XlaOp RngNormal(const XlaOp& mu, const XlaOp& sigma, const Shape& shape);

  // Enqueues a U(a, b) random number generation instruction onto the
  // computation. Returns values in the semi-open interval [a, b).
  XlaOp RngUniform(const XlaOp& a, const XlaOp& b, const Shape& shape);

  // Enqueues a while node onto the computation.
  XlaOp While(const XlaComputation& condition, const XlaComputation& body,
              const XlaOp& init);

  // Enqueues a conditional node onto the computation.
  XlaOp Conditional(const XlaOp& predicate, const XlaOp& true_operand,
                    const XlaComputation& true_computation,
                    const XlaOp& false_operand,
                    const XlaComputation& false_computation);

  // Enqueues a ReducePrecision node onto the computation.
  XlaOp ReducePrecision(const XlaOp& operand, const int exponent_bits,
                        const int mantissa_bits);

  // Enqueues a Gather node onto the computation.
  XlaOp Gather(const XlaOp& input, const XlaOp& gather_indices,
               const GatherDimensionNumbers& dimension_numbers,
               tensorflow::gtl::ArraySlice<int64> window_bounds);

  // Enqueues a Send node onto the computation for device-to-device
  // communication, to send the given operand to a Recv instruction that shares
  // the same channel handle.
  void Send(const XlaOp& operand, const ChannelHandle& handle);
  XlaOp SendWithToken(const XlaOp& operand, const XlaOp& token,
                      const ChannelHandle& handle);

  // Enqueues a Send node which sends data to the host.
  XlaOp SendToHost(const XlaOp& operand, const XlaOp& token,
                   const Shape& shape_with_layout, const ChannelHandle& handle);

  // Enqueues a Recv node which receives data from the host.
  XlaOp RecvFromHost(const XlaOp& token, const Shape& shape,
                     const ChannelHandle& handle);

  // Enqueues an AfterAll operation with no operands producing a token-shaped
  // value.
  XlaOp CreateToken();

  // Enqueues an AfterAll operation with no operands producing a token-shaped
  // value.
  XlaOp AfterAll(tensorflow::gtl::ArraySlice<XlaOp> tokens);

  // Enqueues a Recv node onto the computation. The data comes from a Send
  // instruction that shares the same channel handle and its shape must
  // be the same as the given shape.
  XlaOp Recv(const Shape& shape, const ChannelHandle& handle);
  XlaOp RecvWithToken(const XlaOp& token, const Shape& shape,
                      const ChannelHandle& handle);

  // Normalizes operand across spatial and batch dimensions for each feature.
  //
  // Returns a tuple (normalized, batch_mean, batch_var) where `normalized`
  // is the normalized result and batch_mean and batch_var are the mean and
  // variance, respectively, across batch for the operand.
  XlaOp BatchNormTraining(const XlaOp& operand, const XlaOp& scale,
                          const XlaOp& offset, float epsilon,
                          int64 feature_index);

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
  XlaOp BatchNormInference(const XlaOp& operand, const XlaOp& scale,
                           const XlaOp& offset, const XlaOp& mean,
                           const XlaOp& variance, float epsilon,
                           int64 feature_index);

  // Calculates the gradients of a batch norm op.
  //
  // The inputs `batch_mean` and `batch_var` represent the mean and variance
  // across the batch.
  //
  // Returns a tuple of three elements:
  //   - grad_operand: Gradient with respect to input `operand`
  //   - grad_offset: Gradient with respect to input `offset`
  //   - grad_scale: Gradient with respect to input `scale`
  XlaOp BatchNormGrad(const XlaOp& operand, const XlaOp& scale,
                      const XlaOp& batch_mean, const XlaOp& batch_var,
                      const XlaOp& grad_output, float epsilon,
                      int64 feature_index);

  StatusOr<XlaOp> AddInstruction(
      HloInstructionProto&& instr, HloOpcode opcode,
      tensorflow::gtl::ArraySlice<XlaOp> operands = {});

  void AddCalledComputation(const XlaComputation& computation,
                            HloInstructionProto* instr);

  StatusOr<const HloInstructionProto*> LookUpInstruction(const XlaOp& op) const;

  // Internal helper method that does the building for an arbitrary unary op.
  XlaOp UnaryOp(HloOpcode unop, const XlaOp& operand);

  // Internal helper method that does the building for an arbitrary binary op.
  // broadcast_dimensions specifies which dimensions to use for broadcasting
  // when the operation is between tensors of different ranks.
  XlaOp BinaryOp(HloOpcode binop, const XlaOp& lhs, const XlaOp& rhs,
                 tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  // Internal helper method that does the building for an arbitrary ternary op.
  XlaOp TernaryOp(HloOpcode triop, const XlaOp& lhs, const XlaOp& rhs,
                  const XlaOp& ehs);

  XlaOp RngOp(RandomDistribution distribution,
              tensorflow::gtl::ArraySlice<XlaOp> parameters,
              const Shape& shape);

  StatusOr<XlaOp> InDimBroadcast(
      const Shape& shape, const XlaOp& operand,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  // Internal helper method that creates a sequence of instructions that
  // performs an explicit broadcast of the operand to the target shape.
  StatusOr<XlaOp> AddBroadcastSequence(const Shape& output_shape,
                                       const XlaOp& operand);

  // Internal helper method for creating a Reshape op with the already inferred
  // shape.
  StatusOr<XlaOp> Reshape(const Shape& shape, const XlaOp& operand);

  // Returns the (inferred) result for the program shape for the current
  // computation and fills the root_id in the pointer.
  StatusOr<ProgramShape> GetProgramShape(int64* root_id) const;

  // Returns shapes for the operands.
  StatusOr<std::vector<Shape>> GetOperandShapes(
      tensorflow::gtl::ArraySlice<XlaOp> operands) const;

  // A visitor which checks whether an operation is a compile-time constant,
  // meaning that it doesn't depend on any parameters, or on any stateful
  // operation such as `RngNormal` or `Infeed`. The visitor walks the
  // computation starting at a given operation and sets is_constant to false iff
  // a parameter or stateful operation is encountered.
  void IsConstantVisitor(const int64 op_handle, std::set<int64>* visited,
                         bool* is_constant) const;

  // Checks bounds for convolution parameters.
  Status VerifyConvolution(
      const Shape& lhs_shape, const Shape& rhs_shape,
      const ConvolutionDimensionNumbers& dimension_numbers) const;

  // Helper function for creating a Window proto from user-supplied data.
  // Returns error if the user-supplied data was invalid.
  StatusOr<Window> MakeWindow(
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      tensorflow::gtl::ArraySlice<int64> lhs_dilation,
      tensorflow::gtl::ArraySlice<int64> rhs_dilation) const;

  string name_;  // Name to use for the built computation.

  // The first error encountered while building the computation.
  // This is OK until the first error is encountered.
  Status first_error_;

  // The saved stack trace from the point at which the first error occurred.
  tensorflow::SavedStackTrace first_error_backtrace_;

  // The instructions of this computation.
  std::vector<HloInstructionProto> instructions_;

  // The embedded computations used by this computation. Each computation was
  // the entry computation of some XlaComputation, the key is the unique id of
  // that XlaComputation.
  std::map<int64, HloComputationProto> embedded_;

  // The unique parameter numbers.
  tensorflow::gtl::FlatSet<int64> parameter_numbers_;

  // The metadata to attach to each op. This is structured as a "modal"-like
  // operation, in order to simplify client code (and not sprinkle this metadata
  // throughout the TensorFlow op kernel implementations).
  OpMetadata metadata_;

  // Sharding for this operator. This is structured as a "model"-like operation,
  // in order to simplify client code, similar to metadata_.
  tensorflow::gtl::optional<OpSharding> sharding_;

  // Mode bit that indicates whether to die when a first error is encountered.
  bool die_immediately_on_error_ = false;

  XlaBuilder* parent_builder_{nullptr};

  friend XlaOp Parameter(XlaBuilder* builder, int64 parameter_number,
                         const Shape& shape, const string& name);
  friend XlaOp ConstantLiteral(XlaBuilder* builder,
                               const LiteralSlice& literal);
  template <typename NativeT>
  friend XlaOp ConstantR0(XlaBuilder* builder, NativeT value);
  template <typename NativeT>
  friend XlaOp ConstantR1(XlaBuilder* builder,
                          tensorflow::gtl::ArraySlice<NativeT> values);
  friend XlaOp ConstantR1(XlaBuilder* builder,
                          const tensorflow::core::Bitmap& values);
  template <typename NativeT>
  friend XlaOp ConstantR2(
      XlaBuilder* builder,
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  friend XlaOp ConstantFromArrayWithLayout(XlaBuilder* builder,
                                           const Array<NativeT>& values,
                                           const Layout& layout);
  template <typename NativeT>
  friend XlaOp ConstantFromArray(XlaBuilder* builder,
                                 const Array<NativeT>& values);
  template <typename NativeT>
  friend XlaOp ConstantR2FromArray2DWithLayout(XlaBuilder* builder,
                                               const Array2D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  friend XlaOp ConstantR2FromArray2D(XlaBuilder* builder,
                                     const Array2D<NativeT>& values);
  template <typename NativeT>
  friend XlaOp ConstantR3FromArray3DWithLayout(XlaBuilder* builder,
                                               const Array3D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  friend XlaOp ConstantR3FromArray3D(XlaBuilder* builder,
                                     const Array3D<NativeT>& values);
  template <typename NativeT>
  friend XlaOp ConstantR4FromArray4DWithLayout(XlaBuilder* builder,
                                               const Array4D<NativeT>& values,
                                               const Layout& layout);
  template <typename NativeT>
  friend XlaOp ConstantR4FromArray4D(XlaBuilder* builder,
                                     const Array4D<NativeT>& values);

  template <typename NativeT>
  friend XlaOp ConstantR1(XlaBuilder* builder, int64 length, NativeT value);

  friend XlaOp Broadcast(const XlaOp& operand,
                         tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

  friend XlaOp BroadcastInDim(
      const XlaOp& operand, const Shape& shape,
      const tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  friend XlaOp Pad(const XlaOp& operand, const XlaOp& padding_value,
                   const PaddingConfig& padding_config);

  friend XlaOp Reshape(const XlaOp& operand,
                       tensorflow::gtl::ArraySlice<int64> dimensions,
                       tensorflow::gtl::ArraySlice<int64> new_sizes);

  friend XlaOp Reshape(const XlaOp& operand,
                       tensorflow::gtl::ArraySlice<int64> new_sizes);

  friend XlaOp Collapse(const XlaOp& operand,
                        tensorflow::gtl::ArraySlice<int64> dimensions);

  friend XlaOp Slice(const XlaOp& operand,
                     tensorflow::gtl::ArraySlice<int64> start_indices,
                     tensorflow::gtl::ArraySlice<int64> limit_indices,
                     tensorflow::gtl::ArraySlice<int64> strides);

  friend XlaOp SliceInDim(const XlaOp& operand, int64 start_index,
                          int64 limit_index, int64 stride, int64 dimno);

  friend XlaOp DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                            tensorflow::gtl::ArraySlice<int64> slice_sizes);

  friend XlaOp DynamicUpdateSlice(const XlaOp& operand, const XlaOp& update,
                                  const XlaOp& start_indices);

  friend XlaOp ConcatInDim(XlaBuilder* builder,
                           tensorflow::gtl::ArraySlice<XlaOp> operands,
                           int64 dimension);

  friend void Trace(const string& tag, const XlaOp& operand);

  friend XlaOp Select(const XlaOp& pred, const XlaOp& on_true,
                      const XlaOp& on_false);
  friend XlaOp Tuple(XlaBuilder* builder,
                     tensorflow::gtl::ArraySlice<XlaOp> elements);
  friend XlaOp GetTupleElement(const XlaOp& tuple_data, int64 index);
  friend XlaOp Eq(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Ne(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Ge(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Gt(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Lt(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Le(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Dot(const XlaOp& lhs, const XlaOp& rhs);
  friend XlaOp DotGeneral(const XlaOp& lhs, const XlaOp& rhs,
                          const DotDimensionNumbers& dimension_numbers);
  friend XlaOp Conv(const XlaOp& lhs, const XlaOp& rhs,
                    tensorflow::gtl::ArraySlice<int64> window_strides,
                    Padding padding);
  friend XlaOp ConvWithGeneralPadding(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);
  friend XlaOp ConvWithGeneralDimensions(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
      const ConvolutionDimensionNumbers& dimension_numbers);
  friend XlaOp ConvGeneral(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      const ConvolutionDimensionNumbers& dimension_numbers);
  friend XlaOp ConvGeneralDilated(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      tensorflow::gtl::ArraySlice<int64> lhs_dilation,
      tensorflow::gtl::ArraySlice<int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers);
  friend XlaOp Fft(const XlaOp& operand, FftType fft_type,
                   tensorflow::gtl::ArraySlice<int64> fft_length);
  friend XlaOp Infeed(XlaBuilder* builder, const Shape& shape,
                      const string& config);
  friend void Outfeed(const XlaOp& operand, const Shape& shape_with_layout,
                      const string& outfeed_config);
  friend XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
                    tensorflow::gtl::ArraySlice<XlaOp> operands);
  friend XlaOp CustomCall(XlaBuilder* builder, const string& call_target_name,
                          tensorflow::gtl::ArraySlice<XlaOp> operands,
                          const Shape& shape);
  friend XlaOp HostCompute(XlaBuilder* builder,
                           tensorflow::gtl::ArraySlice<XlaOp> operands,
                           const string& channel_name, int64 cost_estimate_ns,
                           const Shape& shape);
  friend XlaOp Complex(const XlaOp& real, const XlaOp& imag,
                       tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Conj(const XlaOp& operand);
  friend XlaOp Add(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Sub(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Mul(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Div(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Rem(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Max(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Min(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp And(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Or(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Xor(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Not(const XlaOp& operand);
  friend XlaOp ShiftLeft(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp ShiftRightArithmetic(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp ShiftRightLogical(
      const XlaOp& lhs, const XlaOp& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Reduce(const XlaOp& operand, const XlaOp& init_value,
                      const XlaComputation& computation,
                      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce);
  friend XlaOp ReduceAll(const XlaOp& operand, const XlaOp& init_value,
                         const XlaComputation& computation);
  friend XlaOp ReduceWindow(
      const XlaOp& operand, const XlaOp& init_value,
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding);
  friend XlaOp ReduceWindowWithGeneralPadding(
      const XlaOp& operand, const XlaOp& init_value,
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);
  friend XlaOp CrossReplicaSum(
      const XlaOp& operand,
      tensorflow::gtl::ArraySlice<int64> replica_group_ids);
  friend XlaOp CrossReplicaSum(
      const XlaOp& operand, const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<int64> replica_group_ids,
      const tensorflow::gtl::optional<ChannelHandle>& channel_id);
  friend XlaOp SelectAndScatter(
      const XlaOp& operand, const XlaComputation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
      const XlaOp& source, const XlaOp& init_value,
      const XlaComputation& scatter);
  friend XlaOp SelectAndScatterWithGeneralPadding(
      const XlaOp& operand, const XlaComputation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      const XlaOp& source, const XlaOp& init_value,
      const XlaComputation& scatter);
  friend XlaOp Abs(const XlaOp& operand);
  friend XlaOp Atan2(const XlaOp& y, const XlaOp& x,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp Exp(const XlaOp& operand);
  friend XlaOp Expm1(const XlaOp& operand);
  friend XlaOp Floor(const XlaOp& operand);
  friend XlaOp Ceil(const XlaOp& operand);
  friend XlaOp Round(const XlaOp& operand);
  friend XlaOp Log(const XlaOp& operand);
  friend XlaOp Log1p(const XlaOp& operand);
  friend XlaOp Sign(const XlaOp& operand);
  friend XlaOp Clz(const XlaOp& operand);
  friend XlaOp Cos(const XlaOp& operand);
  friend XlaOp Sin(const XlaOp& operand);
  friend XlaOp Tanh(const XlaOp& operand);
  friend XlaOp Real(const XlaOp& operand);
  friend XlaOp Imag(const XlaOp& operand);
  friend XlaOp Pow(const XlaOp& lhs, const XlaOp& rhs,
                   tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);
  friend XlaOp IsFinite(const XlaOp& operand);
  // TODO(b/64798317): Finish CPU & GPU implementation, then replace xla::Iota
  // in xla/client/lib/numeric.h with this (renamed to xla::Iota).
  friend XlaOp IotaGen(XlaBuilder* builder, PrimitiveType type, int64 size);
  friend XlaOp ConvertElementType(const XlaOp& operand,
                                  PrimitiveType new_element_type);
  friend XlaOp BitcastConvertType(const XlaOp& operand,
                                  PrimitiveType new_element_type);
  friend XlaOp Neg(const XlaOp& operand);
  friend XlaOp Transpose(const XlaOp& operand,
                         tensorflow::gtl::ArraySlice<int64> permutation);
  friend XlaOp Rev(const XlaOp& operand,
                   tensorflow::gtl::ArraySlice<int64> dimensions);
  friend XlaOp Sort(XlaOp keys, tensorflow::gtl::optional<XlaOp> values,
                    int64 dimension);
  friend XlaOp Clamp(const XlaOp& min, const XlaOp& operand, const XlaOp& max);
  friend XlaOp Map(XlaBuilder* builder,
                   tensorflow::gtl::ArraySlice<XlaOp> operands,
                   const XlaComputation& computation,
                   tensorflow::gtl::ArraySlice<int64> dimensions,
                   tensorflow::gtl::ArraySlice<XlaOp> static_operands);
  friend XlaOp RngNormal(const XlaOp& mu, const XlaOp& sigma,
                         const Shape& shape);
  friend XlaOp RngUniform(const XlaOp& a, const XlaOp& b, const Shape& shape);
  friend XlaOp While(const XlaComputation& condition,
                     const XlaComputation& body, const XlaOp& init);
  friend XlaOp Conditional(const XlaOp& predicate, const XlaOp& true_operand,
                           const XlaComputation& true_computation,
                           const XlaOp& false_operand,
                           const XlaComputation& false_computation);
  friend XlaOp ReducePrecision(const XlaOp& operand, const int exponent_bits,
                               const int mantissa_bits);
  friend XlaOp Gather(const XlaOp& input, const XlaOp& gather_indices,
                      const GatherDimensionNumbers& dimension_numbers,
                      tensorflow::gtl::ArraySlice<int64> window_bounds);
  friend void Send(const XlaOp& operand, const ChannelHandle& handle);
  friend XlaOp Recv(XlaBuilder* builder, const Shape& shape,
                    const ChannelHandle& handle);
  friend XlaOp BatchNormTraining(const XlaOp& operand, const XlaOp& scale,
                                 const XlaOp& offset, float epsilon,
                                 int64 feature_index);
  friend XlaOp BatchNormInference(const XlaOp& operand, const XlaOp& scale,
                                  const XlaOp& offset, const XlaOp& mean,
                                  const XlaOp& variance, float epsilon,
                                  int64 feature_index);
  friend XlaOp BatchNormGrad(const XlaOp& operand, const XlaOp& scale,
                             const XlaOp& batch_mean, const XlaOp& batch_var,
                             const XlaOp& grad_output, float epsilon,
                             int64 feature_index);
  friend XlaOp SendWithToken(const XlaOp& operand, const XlaOp& token,
                             const ChannelHandle& handle);
  friend XlaOp RecvWithToken(const XlaOp& token, const Shape& shape,
                             const ChannelHandle& handle);
  friend XlaOp SendToHost(const XlaOp& operand, const XlaOp& token,
                          const Shape& shape_with_layout,
                          const ChannelHandle& handle);
  friend XlaOp RecvFromHost(const XlaOp& token, const Shape& shape,
                            const ChannelHandle& handle);
  friend XlaOp InfeedWithToken(const XlaOp& token, const Shape& shape,
                               const string& config);
  friend XlaOp OutfeedWithToken(const XlaOp& operand, const XlaOp& token,
                                const Shape& shape_with_layout,
                                const string& outfeed_config);
  friend XlaOp CreateToken(XlaBuilder* builder);
  friend XlaOp AfterAll(XlaBuilder* builder,
                        tensorflow::gtl::ArraySlice<XlaOp> tokens);
};

// RAII-style object: sets the current sharding assignment in builder on
// construction, and sets back to the previous assignment on destruction.
class XlaScopedShardingAssignment {
 public:
  XlaScopedShardingAssignment(xla::XlaBuilder* builder,
                              tensorflow::gtl::optional<OpSharding> sharding)
      : builder_(builder), prev_sharding_(builder->sharding()) {
    SetSharding(sharding);
  }

  XlaScopedShardingAssignment(const XlaScopedShardingAssignment&) = delete;
  XlaScopedShardingAssignment& operator=(const XlaScopedShardingAssignment&) =
      delete;

  ~XlaScopedShardingAssignment() { SetSharding(prev_sharding_); }

 private:
  void SetSharding(const tensorflow::gtl::optional<OpSharding>& sharding) {
    if (sharding.has_value()) {
      builder_->SetSharding(sharding.value());
    } else {
      builder_->ClearSharding();
    }
  }

  xla::XlaBuilder* const builder_;
  tensorflow::gtl::optional<OpSharding> prev_sharding_;
};

// Free functions for building XlaOps. The intention is that these will
// become the public API for building XlaOps rather than calling methods on
// XlaBuilder directly.

// Enqueues a "retrieve parameter value" instruction for a parameter that was
// passed to the computation.
XlaOp Parameter(XlaBuilder* builder, int64 parameter_number, const Shape& shape,
                const string& name);

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
//   int32          S32
//   int64          S64
//   uint32         U32
//   uint64         U64
//   float          F32
//   double         F64
//
// Note: not all primitive types defined in xla_data.proto have a
// corresponding native type yet.
template <typename NativeT>
XlaOp ConstantR0(XlaBuilder* builder, NativeT value);
template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder,
                 tensorflow::gtl::ArraySlice<NativeT> values);
XlaOp ConstantR1(XlaBuilder* builder, const tensorflow::core::Bitmap& values);
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
XlaOp ConstantR1(XlaBuilder* builder, int64 length, NativeT value);

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
XlaOp Broadcast(const XlaOp& operand,
                tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

// Performs in-dimension-style broadcast.
//
// Operand specifies the input to be broadcast. "shape" is expected output
// shape. "broadcast_dimensions" are the dimensions to be broadcasting into.
// Dimension numbers in broadcast_dimensions map to individual dimensions
// of the operand, and specify what dimension of the output shape they
// should be broadcast.
// e.g.
// Say operand = [1, 2], i.e., a 1D tensor with 2 elements.
// and dimension of shape is [2,2].
// Specifying {1} as brodcast_dimension will generate output
// [1 , 2]
// [1 , 2]
// On the other hand, specifying {0} as broadcast_dimension
// will generate output
// [1 , 1]
// [2 , 2]
XlaOp BroadcastInDim(
    const XlaOp& operand, const Shape& shape,
    const tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

// Enqueues a pad operation onto the computation that pads the given value on
// the edges as well as between the elements of the input. padding_config
// specifies the padding amount for each dimension.
XlaOp Pad(const XlaOp& operand, const XlaOp& padding_value,
          const PaddingConfig& padding_config);

// Enqueues an operation onto the computation that flattens the operand based
// on the dimension order (major/slowest-varying to minor/fastest-varying)
// given, followed by reshaping it into the shape with the given dimension
// sizes (also major to minor). Conceptually, this is a limited form of
// "shape casting".
XlaOp Reshape(const XlaOp& operand,
              tensorflow::gtl::ArraySlice<int64> dimensions,
              tensorflow::gtl::ArraySlice<int64> new_sizes);

// Enqueues an operation onto the computation that collapses the operand, from
// first to last dimension (C order), then reshapes it to the given dimension
// sizes. Conceptually, this is a limited form of "shape casting".
XlaOp Reshape(const XlaOp& operand,
              tensorflow::gtl::ArraySlice<int64> new_sizes);

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
XlaOp Collapse(const XlaOp& operand,
               tensorflow::gtl::ArraySlice<int64> dimensions);

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
XlaOp Slice(const XlaOp& operand,
            tensorflow::gtl::ArraySlice<int64> start_indices,
            tensorflow::gtl::ArraySlice<int64> limit_indices,
            tensorflow::gtl::ArraySlice<int64> strides);

// Enqueues a slice operation in a given dimension, taking all other
// dimensions as they are; e.g. if dimno is 1 from start_index 2 to
// limit_index 4 by 1, and the shape is f32[7,8,9], this call is short-hand
// for:
//
//  array[:, 2:4:1, :]
XlaOp SliceInDim(const XlaOp& operand, int64 start_index, int64 limit_index,
                 int64 stride, int64 dimno);

// Enqueues a slice operation onto the computation that slices the 'operand'
// from dynamic start indices which are passed in 'start_indices'.
// The size of the slice in each dimension is passed in 'slice_sizes',
// which specify the end point of exclusive slice intervals in each
// dimension [start, start + size).
// The shape of 'start_indices' must be rank == 1, with dimension size
// equal to the rank of the 'operand'.
// Slice index calculations are computed modulo input dimension sizes to
// prevent dynamic start indices from generating out-of-bound array accesses.
XlaOp DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                   tensorflow::gtl::ArraySlice<int64> slice_sizes);

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
// The shape of 'start_indices' must be rank == 1, with dimension size
// equal to the rank of the 'operand'.
// Slice index calculations are computed modulo update dimension sizes to
// prevent dynamic start indices from generating out-of-bound array accesses.
XlaOp DynamicUpdateSlice(const XlaOp& operand, const XlaOp& update,
                         const XlaOp& start_indices);

// Enqueues a concatenate instruction onto the computation. 'operands' must
// have >= 1 entry.
XlaOp ConcatInDim(XlaBuilder* builder,
                  tensorflow::gtl::ArraySlice<XlaOp> operands, int64 dimension);

// Enqueue a tracing operation onto the computation; the computation will emit
// a logging message with the operand.
void Trace(const string& tag, const XlaOp& operand);

// Enqueues a conditional-move-like select operation onto the computation;
// predicated on pred, selects between on_true and on_false.
XlaOp Select(const XlaOp& pred, const XlaOp& on_true, const XlaOp& on_false);

// Enqueues a tuple-creation instruction onto the computation.
XlaOp Tuple(XlaBuilder* builder, tensorflow::gtl::ArraySlice<XlaOp> elements);

// Enqueues a tuple-element-get instruction onto the computation.
XlaOp GetTupleElement(const XlaOp& tuple_data, int64 index);

// Enqueues an equal-to comparison instruction onto the computation.
XlaOp Eq(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a not-equal comparison instruction onto the computation.
XlaOp Ne(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a greater-or-equal comparison instruction onto the computation.
XlaOp Ge(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a greater-than comparison instruction onto the computation.
XlaOp Gt(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a less-than comparison instruction onto the computation.
XlaOp Lt(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a less-or-equal comparison instruction onto the computation.
XlaOp Le(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a dot instruction onto the computation.
XlaOp Dot(const XlaOp& lhs, const XlaOp& rhs);

// Enqueues a general dot instruction onto the computation.
XlaOp DotGeneral(const XlaOp& lhs, const XlaOp& rhs,
                 const DotDimensionNumbers& dimension_numbers);

// Enqueues a convolution instruction onto the computation, which uses the
// default convolution dimension numbers.
XlaOp Conv(const XlaOp& lhs, const XlaOp& rhs,
           tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding);

// Enqueues a convolution instruction onto the computation, with the caller
// provided padding configuration in the format returned by MakePadding().
XlaOp ConvWithGeneralPadding(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

// Enqueues a convolution instruction onto the computation, with the caller
// provided dimension numbers configuration.
XlaOp ConvWithGeneralDimensions(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const ConvolutionDimensionNumbers& dimension_numbers);

// Enqueues a convolution instruction onto the computation, with the caller
// provided padding configuration as well as the dimension numbers.
XlaOp ConvGeneral(const XlaOp& lhs, const XlaOp& rhs,
                  tensorflow::gtl::ArraySlice<int64> window_strides,
                  tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
                  const ConvolutionDimensionNumbers& dimension_numbers);

// Enqueues a convolution instruction onto the computation, with the caller
// provided padding configuration, dilation factors and dimension numbers.
XlaOp ConvGeneralDilated(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers);

// Enqueues an FFT instruction onto the computation, of the given type and
// with the given FFT length.
XlaOp Fft(const XlaOp& operand, FftType fft_type,
          tensorflow::gtl::ArraySlice<int64> fft_length);

// Enqueues an infeed instruction onto the computation, which writes data of
// the given shape to the infeed buffer of the device.
XlaOp Infeed(XlaBuilder* builder, const Shape& shape,
             const string& config = "");

// Variant of Infeed which takes a token-shaped operand and produces a
// two-element tuple containing the data value and a token-shaped value.
// Tokens are used for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp InfeedWithToken(const XlaOp& token, const Shape& shape,
                      const string& config = "");

// Enqueues an outfeed instruction onto the computation. This instruction
// generates outgoing data transfers for the given data.
//
// shape_with_layout communicates the laid out shape that we want to outfeed
// -- if !ShapeUtil::Compatible(GetShape(operand), shape_with_layout) an error
// will occur.
void Outfeed(const XlaOp& operand, const Shape& shape_with_layout,
             const string& outfeed_config);

// Variant of Outfeed which takes a token-shaped operand and produces a
// token-shaped value. Tokens are used for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp OutfeedWithToken(const XlaOp& operand, const XlaOp& token,
                       const Shape& shape_with_layout,
                       const string& outfeed_config);

// Enqueues a call instruction onto the computation.
XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
           tensorflow::gtl::ArraySlice<XlaOp> operands);

// Enqueues a custom call instruction onto the computation.
// During code generation, a call instruction is emitted which targets a
// symbol with the name |call_target_name|.  The |operands| are passed to the
// call instruction.  |shape| is the resultant shape.
XlaOp CustomCall(XlaBuilder* builder, const string& call_target_name,
                 tensorflow::gtl::ArraySlice<XlaOp> operands,
                 const Shape& shape);

// Enqueues a pseudo-op to represent host-side computation data-dependencies.
// During code generation, host send and receive operations will be generated
// to transfer |operands| to the host and a single result of |shape| back to
// the device.  Host send/recv operations are emitted using |channel_name|.
// Dataflow dependencies and the |cost_estimate_ns| field may be used in HLO
// instruction scheduling.
XlaOp HostCompute(XlaBuilder* builder,
                  tensorflow::gtl::ArraySlice<XlaOp> operands,
                  const string& channel_name, int64 cost_estimate_ns,
                  const Shape& shape);

// The following methods enqueue element-wise binary arithmetic operations
// onto the computation. The shapes of the operands have to match unless one
// of the operands is a scalar, or an explicit broadcast dimension is given
// (see g3doc for more details).

// Enqueues a complex compose instruction onto the computation.
XlaOp Complex(const XlaOp& real, const XlaOp& imag,
              tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a complex conjugate instruction onto the computation.
XlaOp Conj(const XlaOp& operand);

// Enqueues an add instruction onto the computation.
XlaOp Add(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a subtract instruction onto the computation.
XlaOp Sub(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a multiply instruction onto the computation.
XlaOp Mul(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a divide instruction onto the computation.
XlaOp Div(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a remainder instruction onto the computation.
XlaOp Rem(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a max instruction onto the computation.
XlaOp Max(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues a min instruction onto the computation.
XlaOp Min(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Element-wise logical operators
XlaOp And(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

XlaOp Or(const XlaOp& lhs, const XlaOp& rhs,
         tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

XlaOp Xor(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

XlaOp Not(const XlaOp& operand);

XlaOp ShiftLeft(const XlaOp& lhs, const XlaOp& rhs,
                tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});
XlaOp ShiftRightArithmetic(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});
XlaOp ShiftRightLogical(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Reduces an array among the provided dimensions, given "computation" as a
// reduction operator.
XlaOp Reduce(const XlaOp& operand, const XlaOp& init_value,
             const XlaComputation& computation,
             tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce);

// Convenience wrapper around the above that reduces all the dimensions in the
// operand shape.
XlaOp ReduceAll(const XlaOp& operand, const XlaOp& init_value,
                const XlaComputation& computation);

// Enqueues a windowed reduce instruction onto the computation.
XlaOp ReduceWindow(const XlaOp& operand, const XlaOp& init_value,
                   const XlaComputation& computation,
                   tensorflow::gtl::ArraySlice<int64> window_dimensions,
                   tensorflow::gtl::ArraySlice<int64> window_strides,
                   Padding padding);

// As ReduceWindow(), but the padding is given in the format
// returned by MakePadding().
XlaOp ReduceWindowWithGeneralPadding(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

// Returns the sum of the operand value within each subgroup of replicas. All
// replicas supply one input to the sum and all replicas receive the resulting
// sum for each subgroup.
XlaOp CrossReplicaSum(
    const XlaOp& operand,
    tensorflow::gtl::ArraySlice<int64> replica_group_ids = {});

// Enqueues an operation that do an AllReduce of the operand cross cores. Here
// AllReduce means doing a reduction on the input operand cross cores and then
// broadcasting the reduction result to those cores. The reduction function is
// defined by `computation`, which should be a commutative computation on
// scalars, e.g., add, min, or max. The way that AllReduce is applied is
// configured by:
//
// - `replica_group_ids`: maps replica ids to subgroup ids. If empty, all
// replicas belong to one group. Allreduce will be applied within subgroups.
// For example, we have 4 replicas, then replica_group_ids={0,1,0,1} means,
// replica 0 and 2 are in subgroup 0, replica 1 and 3 are in subgroup 1.
//
// - `channel_id`: for Allreduce nodes from different models, if they have the
// same channel_id, they will be 'Allreduce'd. If empty, Allreduce will not be
// applied cross models.
//
// TODO(b/79737069): Rename this to AllReduce when it's ready to use.
XlaOp CrossReplicaSum(const XlaOp& operand, const XlaComputation& computation,
                      tensorflow::gtl::ArraySlice<int64> replica_group_ids = {},
                      const tensorflow::gtl::optional<ChannelHandle>&
                          channel_id = tensorflow::gtl::nullopt);

// Enqueues an operation that scatters the `source` array to the selected
// indices of each window.
XlaOp SelectAndScatter(const XlaOp& operand, const XlaComputation& select,
                       tensorflow::gtl::ArraySlice<int64> window_dimensions,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       Padding padding, const XlaOp& source,
                       const XlaOp& init_value, const XlaComputation& scatter);

// As SelectAndScatter(), but the padding is given in the format
// returned by MakePadding().
XlaOp SelectAndScatterWithGeneralPadding(
    const XlaOp& operand, const XlaComputation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const XlaOp& source, const XlaOp& init_value,
    const XlaComputation& scatter);

// Enqueues an abs instruction onto the computation.
XlaOp Abs(const XlaOp& operand);

// Enqueues a atan2 instruction onto the computation.
XlaOp Atan2(const XlaOp& y, const XlaOp& x,
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues an exp instruction onto the computation.
XlaOp Exp(const XlaOp& operand);

// Enqueues an expm1 instruction onto the computation.
XlaOp Expm1(const XlaOp& operand);

// Enqueues a floor instruction onto the computation.
XlaOp Floor(const XlaOp& operand);

// Enqueues a ceil instruction onto the computation.
XlaOp Ceil(const XlaOp& operand);

// Enqueues a round instruction onto the computation, rounding to nearest even
// with half-way cases rounding away from zero.
XlaOp Round(const XlaOp& operand);

// Enqueues an log instruction (natural logarithm) onto the computation.
XlaOp Log(const XlaOp& operand);

// Enqueues an log1p instruction (log(x+1)) onto the computation.
XlaOp Log1p(const XlaOp& operand);

// Enqueues a sign instruction onto the computation.
XlaOp Sign(const XlaOp& operand);

// Enqueues a count leading zeros instruction onto the computation.
XlaOp Clz(const XlaOp& operand);

// Enqueues a cosine instruction onto the computation.
XlaOp Cos(const XlaOp& operand);

// Enqueues a sine instruction onto the computation.
XlaOp Sin(const XlaOp& operand);

// Enqueues a tanh instruction onto the computation.
XlaOp Tanh(const XlaOp& operand);

// Enqueues a real-part instruction onto the computation.
XlaOp Real(const XlaOp& operand);

// Enqueues an imaginary-part instruction onto the computation.
XlaOp Imag(const XlaOp& operand);

// Enqueues a lhs^rhs computation onto the computation.
XlaOp Pow(const XlaOp& lhs, const XlaOp& rhs,
          tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

// Enqueues an operator that tests if the operand's values are finite, i.e.,
// not Inf or NaN. Defined only for floating-point types. Returns an array of
// booleans with the same shape where entries are true iff the corresponding
// entry was NaN.
XlaOp IsFinite(const XlaOp& operand);

// Enqueues a convert instruction onto the computation that changes the
// element type of the operand array to primitive_type.
XlaOp ConvertElementType(const XlaOp& operand, PrimitiveType new_element_type);

// Enqueues a no-op instruction onto the computation that changes
// the element type of the operand array to primitive_type. The
// bit-widths of the source and destination element types must be
// identical.
XlaOp BitcastConvertType(const XlaOp& operand, PrimitiveType new_element_type);

// Enqueues a negate instruction onto the computation.
XlaOp Neg(const XlaOp& operand);

// Enqueues a transpose instruction onto the computation.
XlaOp Transpose(const XlaOp& operand,
                tensorflow::gtl::ArraySlice<int64> permutation);

// Enqueues a reverse instruction onto the computation. The order of the
// elements in the given dimensions is reversed (i.e., the element at index i
// is moved to index dimension_size - 1 - i).
XlaOp Rev(const XlaOp& operand, tensorflow::gtl::ArraySlice<int64> dimensions);

// Enqueues a sort (as increasing order) instruction onto the computation.
// If only keys are provided:
// * If the keys are an rank-1 tensor (an array), the result is a sorted array
// of keys, in ascending order.
// * If the keys have higher rank, the keys are sorted along the provided
// dimension. For example, for a rank-2 tensor (a matrix) of keys, a dimension
// value of 0 will indepenently sort every column, and a dimension value of 1
// will independently sort each row. If no dimension number is provided, then
// the last dimension is chosen by default.
//
// If both keys and values are provided:
// * The keys and the values must tensors with the same dimensions. The
// element types of the tensors may be different.
// * The result is a tuple that consists of a sorted tensor of keys (along the
// provided dimension, as above) as the first element, and a tensor with their
// corresponding values as the second element.
XlaOp Sort(XlaOp keys,
           tensorflow::gtl::optional<XlaOp> values = tensorflow::gtl::nullopt,
           int64 dimension = -1);

// Enqueues a clamp instruction onto the computation.
XlaOp Clamp(const XlaOp& min, const XlaOp& operand, const XlaOp& max);

// Enqueues a map instruction onto the computation.
XlaOp Map(XlaBuilder* builder, tensorflow::gtl::ArraySlice<XlaOp> operands,
          const XlaComputation& computation,
          tensorflow::gtl::ArraySlice<int64> dimensions,
          tensorflow::gtl::ArraySlice<XlaOp> static_operands = {});

// Enqueues a N(mu, sigma) random number generation instruction onto the
// computation.
XlaOp RngNormal(const XlaOp& mu, const XlaOp& sigma, const Shape& shape);

// Enqueues a U(a, b) random number generation instruction onto the
// computation. Returns values in the semi-open interval [a, b).
XlaOp RngUniform(const XlaOp& a, const XlaOp& b, const Shape& shape);

// Enqueues a while node onto the computation.
XlaOp While(const XlaComputation& condition, const XlaComputation& body,
            const XlaOp& init);

// Enqueues a conditional node onto the computation.
XlaOp Conditional(const XlaOp& predicate, const XlaOp& true_operand,
                  const XlaComputation& true_computation,
                  const XlaOp& false_operand,
                  const XlaComputation& false_computation);

// Enqueues a ReducePrecision node onto the computation.
XlaOp ReducePrecision(const XlaOp& operand, const int exponent_bits,
                      const int mantissa_bits);

// Enqueues a Gather node onto the computation.
XlaOp Gather(const XlaOp& input, const XlaOp& gather_indices,
             const GatherDimensionNumbers& dimension_numbers,
             tensorflow::gtl::ArraySlice<int64> window_bounds);

// Enqueues a Send node onto the computation for device-to-device
// communication. This operation sends the given operand to
// a Recv instruction in a different computation that shares the same channel
// handle.
void Send(const XlaOp& operand, const ChannelHandle& handle);

// Variant of Send which takes a token-shaped operand and produces a
// token-shaped value.  Tokens are used for ordering side-effecting operations.
// TODO(b/110532604): Replace all uses of the non-token form with this variant.
XlaOp SendWithToken(const XlaOp& operand, const XlaOp& token,
                    const ChannelHandle& handle);

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
XlaOp RecvWithToken(const XlaOp& token, const Shape& shape,
                    const ChannelHandle& handle);

// Enqueues a Send node which transfers data from the device to the host. The
// 'shape_with_layout' argument defines the layout of the data transferred; its
// shape must be compatible with the shape of the operand. The operand must be
// array-shaped.
// TODO(b/111544877): Support tuple shapes.
XlaOp SendToHost(const XlaOp& operand, const XlaOp& token,
                 const Shape& shape_with_layout, const ChannelHandle& handle);

// Enqueues a Recv node which transfers data from the host to the device. The
// given shape must contain a layout and must be an array.
// TODO(b/111544877): Support tuple shapes.
XlaOp RecvFromHost(const XlaOp& token, const Shape& shape,
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
XlaOp AfterAll(XlaBuilder* builder, tensorflow::gtl::ArraySlice<XlaOp> tokens);

// Normalizes operand across spatial and batch dimensions for each feature.
//
// Returns a tuple (normalized, batch_mean, batch_var) where `normalized`
// is the normalized result and batch_mean and batch_var are the mean and
// variance, respectively, across batch for the operand.
XlaOp BatchNormTraining(const XlaOp& operand, const XlaOp& scale,
                        const XlaOp& offset, float epsilon,
                        int64 feature_index);

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
XlaOp BatchNormInference(const XlaOp& operand, const XlaOp& scale,
                         const XlaOp& offset, const XlaOp& mean,
                         const XlaOp& variance, float epsilon,
                         int64 feature_index);

// Calculates the gradients of a batch norm op.
//
// The inputs `batch_mean` and `batch_var` represent the mean and variance
// across the batch.
//
// Returns a tuple of three elements:
//   - grad_operand: Gradient with respect to input `operand`
//   - grad_offset: Gradient with respect to input `offset`
//   - grad_scale: Gradient with respect to input `scale`
XlaOp BatchNormGrad(const XlaOp& operand, const XlaOp& scale,
                    const XlaOp& batch_mean, const XlaOp& batch_var,
                    const XlaOp& grad_output, float epsilon,
                    int64 feature_index);

// Implementation details below this point.

template <typename NativeT>
XlaOp XlaBuilder::ConstantR0(NativeT value) {
  return ConstantLiteral(*LiteralUtil::CreateR0<NativeT>(value));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR1(tensorflow::gtl::ArraySlice<NativeT> values) {
  return ConstantLiteral(*LiteralUtil::CreateR1<NativeT>(values));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR1(int64 length, NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {length}));
  literal.PopulateWithValue(value);
  return ConstantLiteral(literal);
}

inline XlaOp XlaBuilder::ConstantR1(const tensorflow::core::Bitmap& values) {
  return ConstantLiteral(*LiteralUtil::CreateR1(values));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return ConstantLiteral(*LiteralUtil::CreateR2<NativeT>(values));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantFromArrayWithLayout(const Array<NativeT>& values,
                                              const Layout& layout) {
  return ConstantLiteral(
      *LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantFromArray(const Array<NativeT>& values) {
  return ConstantLiteral(*LiteralUtil::CreateFromArray<NativeT>(values));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout) {
  return ConstantLiteral(
      *LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR2FromArray2D(const Array2D<NativeT>& values) {
  return ConstantLiteral(*LiteralUtil::CreateR2FromArray2D<NativeT>(values));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout) {
  return ConstantLiteral(
      *LiteralUtil::CreateR3FromArray3DWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR3FromArray3D(const Array3D<NativeT>& values) {
  return ConstantFromArray(values);
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout) {
  return ConstantFromArrayWithLayout(values, layout);
}

template <typename NativeT>
XlaOp XlaBuilder::ConstantR4FromArray4D(const Array4D<NativeT>& values) {
  return ConstantFromArray(values);
}

// Free function template implementations.

template <typename NativeT>
XlaOp ConstantR0(XlaBuilder* builder, NativeT value) {
  return ConstantLiteral(builder, *LiteralUtil::CreateR0<NativeT>(value));
}

template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder,
                 tensorflow::gtl::ArraySlice<NativeT> values) {
  return ConstantLiteral(builder, *LiteralUtil::CreateR1<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantR1(XlaBuilder* builder, int64 length, NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {length}));
  literal.PopulateWithValue(value);
  return ConstantLiteral(builder, literal);
}

inline XlaOp ConstantR1(XlaBuilder* builder,
                        const tensorflow::core::Bitmap& values) {
  return ConstantLiteral(builder, *LiteralUtil::CreateR1(values));
}

template <typename NativeT>
XlaOp ConstantR2(XlaBuilder* builder,
                 std::initializer_list<std::initializer_list<NativeT>> values) {
  return ConstantLiteral(builder, *LiteralUtil::CreateR2<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantFromArrayWithLayout(XlaBuilder* builder,
                                  const Array<NativeT>& values,
                                  const Layout& layout) {
  return ConstantLiteral(
      builder,
      *LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp ConstantFromArray(XlaBuilder* builder, const Array<NativeT>& values) {
  return ConstantLiteral(builder,
                         *LiteralUtil::CreateFromArray<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantR2FromArray2DWithLayout(XlaBuilder* builder,
                                      const Array2D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantLiteral(
      builder,
      *LiteralUtil::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
XlaOp ConstantR2FromArray2D(XlaBuilder* builder,
                            const Array2D<NativeT>& values) {
  return ConstantLiteral(builder,
                         *LiteralUtil::CreateR2FromArray2D<NativeT>(values));
}

template <typename NativeT>
XlaOp ConstantR3FromArray3DWithLayout(XlaBuilder* builder,
                                      const Array3D<NativeT>& values,
                                      const Layout& layout) {
  return ConstantLiteral(
      builder,
      *LiteralUtil::CreateR3FromArray3DWithLayout<NativeT>(values, layout));
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

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_BUILDER_H_
