/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_COMPUTATION_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_COMPUTATION_BUILDER_H_

#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Wraps an XLA client with a convenient interface for building up
// computations. Any errors encountered in building up the computation are
// deferred from being handled until Build() is called.
//
// Thread-compatible.
class ComputationBuilder {
 public:
  // client: client in which to build the computation.
  // computation_name: name to use for the built computation.
  ComputationBuilder(Client* client, const string& computation_name);

  ~ComputationBuilder();

  // Returns the client the builder was initialized with.
  Client* client() const { return client_; }

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

  // Enqueues a "retrieve parameter value" instruction for a parameter that was
  // passed to the computation.
  ComputationDataHandle Parameter(int64 parameter_number, const Shape& shape,
                                  const string& name);

  // Retrieves the (inferred) shape of the operand in the computation.
  StatusOr<std::unique_ptr<Shape>> GetShape(
      const ComputationDataHandle& operand);

  // Retrieves the (inferred) result for the current computation's shape.
  StatusOr<ProgramShape> GetProgramShape();

  // Checks that the operand has the given expected shape. Returns the operand
  // if yes, fails with a CHECK error if no.
  ComputationDataHandle CheckShape(const ComputationDataHandle& operand,
                                   const Shape& expected_shape);

  // Checks that the lhs and rhs results have the same shape.
  void CheckSameShape(const ComputationDataHandle& lhs,
                      const ComputationDataHandle& rhs);

  // Enqueues a constant with the value of the given literal onto the
  // computation.
  ComputationDataHandle ConstantLiteral(const Literal& literal);

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
  ComputationDataHandle ConstantR0(NativeT value);
  template <typename NativeT>
  ComputationDataHandle ConstantR1(tensorflow::gtl::ArraySlice<NativeT> values);
  ComputationDataHandle ConstantR1(const tensorflow::core::Bitmap& values);
  template <typename NativeT>
  ComputationDataHandle ConstantR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  ComputationDataHandle ConstantFromArrayWithLayout(
      const Array<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  ComputationDataHandle ConstantFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  ComputationDataHandle ConstantR2FromArray2DWithLayout(
      const Array2D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  ComputationDataHandle ConstantR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  ComputationDataHandle ConstantR3FromArray3DWithLayout(
      const Array3D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  ComputationDataHandle ConstantR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  ComputationDataHandle ConstantR4FromArray4DWithLayout(
      const Array4D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  ComputationDataHandle ConstantR4FromArray4D(const Array4D<NativeT>& values);

  // Enqueues a rank one constant (vector) onto the computation. The vector has
  // size 'length' and every element has the value 'value'.
  template <typename NativeT>
  ComputationDataHandle ConstantR1(int64 length, NativeT value);

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
  ComputationDataHandle Broadcast(
      const ComputationDataHandle& operand,
      tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

  // Enqueues a pad operation onto the computation that pads the given value on
  // the edges as well as between the elements of the input. padding_config
  // specifies the padding amount for each dimension.
  ComputationDataHandle Pad(const ComputationDataHandle& operand,
                            const ComputationDataHandle& padding_value,
                            const PaddingConfig& padding_config);

  // Enqueues an operation onto the computation that flattens the operand based
  // on the dimension order (major/slowest-varying to minor/fastest-varying)
  // given, followed by reshaping it into the shape with the given dimension
  // sizes (also major to minor). Conceptually, this is a limited form of
  // "shape casting".
  ComputationDataHandle Reshape(const ComputationDataHandle& operand,
                                tensorflow::gtl::ArraySlice<int64> dimensions,
                                tensorflow::gtl::ArraySlice<int64> new_sizes);

  // Enqueues an operation onto the computation that collapses the operand, from
  // first to last dimension (C order), then reshapes it to the given dimension
  // sizes. Conceptually, this is a limited form of "shape casting".
  ComputationDataHandle Reshape(const ComputationDataHandle& operand,
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
  ComputationDataHandle Collapse(const ComputationDataHandle& operand,
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
  ComputationDataHandle Slice(const ComputationDataHandle& operand,
                              tensorflow::gtl::ArraySlice<int64> start_indices,
                              tensorflow::gtl::ArraySlice<int64> limit_indices,
                              tensorflow::gtl::ArraySlice<int64> strides);

  // Enqueues a slice operation in a given dimension, taking all other
  // dimensions as they are; e.g. if dimno is 1 from start_index 2 to
  // limit_index 4 by 1, and the shape is f32[7,8,9], this call is short-hand
  // for:
  //
  //  array[:, 2:4:1, :]
  ComputationDataHandle SliceInDim(const ComputationDataHandle& operand,
                                   int64 start_index, int64 limit_index,
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
  ComputationDataHandle DynamicSlice(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& start_indices,
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
  ComputationDataHandle DynamicUpdateSlice(
      const ComputationDataHandle& operand, const ComputationDataHandle& update,
      const ComputationDataHandle& start_indices);

  // Enqueues a concatenate instruction onto the computation. 'operands' must
  // have >= 1 entry.
  ComputationDataHandle ConcatInDim(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
      int64 dimension);

  // Enqueue a tracing operation onto the computation; the computation will emit
  // a logging message with the operand.
  void Trace(const string& tag, const ComputationDataHandle& operand);

  // Enqueues a conditional-move-like select operation onto the computation;
  // predicated on pred, selects between on_true and on_false.
  ComputationDataHandle Select(const ComputationDataHandle& pred,
                               const ComputationDataHandle& on_true,
                               const ComputationDataHandle& on_false);

  // Enqueues a tuple-creation instruction onto the computation.
  ComputationDataHandle Tuple(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> elements);

  // Enqueues a tuple-element-get instruction onto the computation.
  ComputationDataHandle GetTupleElement(const ComputationDataHandle& tuple_data,
                                        int64 index);

  // Enqueues an equal-to comparison instruction onto the computation.
  ComputationDataHandle Eq(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a not-equal comparison instruction onto the computation.
  ComputationDataHandle Ne(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a greater-or-equal comparison instruction onto the computation.
  ComputationDataHandle Ge(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a greater-than comparison instruction onto the computation.
  ComputationDataHandle Gt(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a less-than comparison instruction onto the computation.
  ComputationDataHandle Lt(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a less-or-equal comparison instruction onto the computation.
  ComputationDataHandle Le(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a dot instruction onto the computation.
  ComputationDataHandle Dot(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs);

  // Enqueues a general dot instruction onto the computation.
  ComputationDataHandle DotGeneral(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      const DotDimensionNumbers& dimension_numbers);

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

  // Creates a ConvolutionDimensionNumbers with the given arguments. Returns an
  // error if either the input or the weight dimension numbers have conflicts.
  static StatusOr<ConvolutionDimensionNumbers> CreateConvDimensionNumbers(
      int64 input_batch, int64 input_feature, int64 input_first_spatial,
      int64 input_second_spatial, int64 output_batch, int64 output_feature,
      int64 output_first_spatial, int64 output_second_spatial,
      int64 kernel_output_feature, int64 kernel_input_feature,
      int64 kernel_first_spatial, int64 kernel_second_spatial);

  // Enqueues a convolution instruction onto the computation, which uses the
  // default convolution dimension numbers.
  ComputationDataHandle Conv(const ComputationDataHandle& lhs,
                             const ComputationDataHandle& rhs,
                             tensorflow::gtl::ArraySlice<int64> window_strides,
                             Padding padding);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration in the format returned by MakePadding().
  ComputationDataHandle ConvWithGeneralPadding(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided dimension numbers configuration.
  ComputationDataHandle ConvWithGeneralDimensions(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration as well as the dimension numbers.
  ComputationDataHandle ConvGeneral(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration, dilation factors and dimension numbers.
  ComputationDataHandle ConvGeneralDilated(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      tensorflow::gtl::ArraySlice<int64> lhs_dilation,
      tensorflow::gtl::ArraySlice<int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues an FFT instruction onto the computation, of the given type and
  // with the given FFT length.
  ComputationDataHandle Fft(const ComputationDataHandle& operand,
                            FftType fft_type,
                            tensorflow::gtl::ArraySlice<int64> fft_length);

  // Enqueues an infeed instruction onto the computation, which writes data of
  // the given shape to the infeed buffer of the device.
  ComputationDataHandle Infeed(const Shape& shape, const string& config = "");

  // Enqueues an outfeed instruction onto the computation. This instruction
  // generates outgoing data transfers for the given data.
  //
  // shape_with_layout communicates the laid out shape that we want to outfeed
  // -- if !ShapeUtil::Compatible(GetShape(operand), shape_with_layout) an error
  // will occur.
  void Outfeed(const ComputationDataHandle& operand,
               const Shape& shape_with_layout, const string& outfeed_config);

  // Enqueues a call instruction onto the computation.
  ComputationDataHandle Call(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands);

  // Enqueues a custom call instruction onto the computation.
  // During code generation, a call instruction is emitted which targets a
  // symbol with the name |call_target_name|.  The |operands| are passed to the
  // call instruction.  |shape| is the resultant shape.
  ComputationDataHandle CustomCall(
      const string& call_target_name,
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
      const Shape& shape);

  // Enqueues a pseudo-op to represent host-side computation data-dependencies.
  // During code generation, host send and receive operations will be generated
  // to transfer |operands| to the host and a single result of |shape| back to
  // the device.  Host send/recv operations are emitted using |channel_name|.
  // Dataflow dependencies and the |cost_estimate_ns| field may be used in HLO
  // instruction scheduling.
  ComputationDataHandle HostCompute(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
      const string& channel_name, int64 cost_estimate_ns, const Shape& shape);

  // The following methods enqueue element-wise binary arithmetic operations
  // onto the computation. The shapes of the operands have to match unless one
  // of the operands is a scalar, or an explicit broadcast dimension is given
  // (see g3doc for more details).

  // Enqueues a complex compose instruction onto the computation.
  ComputationDataHandle Complex(
      const ComputationDataHandle& real, const ComputationDataHandle& imag,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a complex conjugate instruction onto the computation.
  ComputationDataHandle Conj(const ComputationDataHandle& operand);

  // Enqueues an add instruction onto the computation.
  ComputationDataHandle Add(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a subtract instruction onto the computation.
  ComputationDataHandle Sub(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a multiply instruction onto the computation.
  ComputationDataHandle Mul(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a divide instruction onto the computation.
  ComputationDataHandle Div(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a remainder instruction onto the computation.
  ComputationDataHandle Rem(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a max instruction onto the computation.
  ComputationDataHandle Max(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a min instruction onto the computation.
  ComputationDataHandle Min(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Element-wise logical operators
  ComputationDataHandle And(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  ComputationDataHandle Or(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  ComputationDataHandle Not(const ComputationDataHandle& operand);

  ComputationDataHandle ShiftLeft(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});
  ComputationDataHandle ShiftRightArithmetic(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});
  ComputationDataHandle ShiftRightLogical(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Reduces an array among the provided dimensions, given "computation" as a
  // reduction operator.
  ComputationDataHandle Reduce(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& init_value, const Computation& computation,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce);

  // Convenience wrapper around the above that reduces all the dimensions in the
  // operand shape.
  ComputationDataHandle ReduceAll(const ComputationDataHandle& operand,
                                  const ComputationDataHandle& init_value,
                                  const Computation& computation);

  // Enqueues a windowed reduce instruction onto the computation.
  ComputationDataHandle ReduceWindow(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& init_value, const Computation& computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding);

  // As ReduceWindow(), but the padding is given in the format
  // returned by MakePadding().
  ComputationDataHandle ReduceWindowWithGeneralPadding(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& init_value, const Computation& computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

  // Returns the sum of the operand value across all replicas. All replicas
  // supply one input to the sum and all replicas receive the resulting sum.
  ComputationDataHandle CrossReplicaSum(const ComputationDataHandle& operand);

  // Enqueues an operation that scatters the `source` array to the selected
  // indices of each window.
  ComputationDataHandle SelectAndScatter(
      const ComputationDataHandle& operand, const Computation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
      const ComputationDataHandle& source,
      const ComputationDataHandle& init_value, const Computation& scatter);

  // As SelectAndScatter(), but the padding is given in the format
  // returned by MakePadding().
  ComputationDataHandle SelectAndScatterWithGeneralPadding(
      const ComputationDataHandle& operand, const Computation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
      const ComputationDataHandle& source,
      const ComputationDataHandle& init_value, const Computation& scatter);

  // Enqueues an abs instruction onto the computation.
  ComputationDataHandle Abs(const ComputationDataHandle& operand);

  // Enqueues a atan2 instruction onto the computation.
  ComputationDataHandle Atan2(
      const ComputationDataHandle& y, const ComputationDataHandle& x,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues an exp instruction onto the computation.
  ComputationDataHandle Exp(const ComputationDataHandle& operand);

  // Enqueues a floor instruction onto the computation.
  ComputationDataHandle Floor(const ComputationDataHandle& operand);

  // Enqueues a ceil instruction onto the computation.
  ComputationDataHandle Ceil(const ComputationDataHandle& operand);

  // Enqueues a round instruction onto the computation, rounding to nearest even
  // with half-way cases rounding away from zero.
  ComputationDataHandle Round(const ComputationDataHandle& operand);

  // Enqueues an log instruction (natural logarithm) onto the computation.
  ComputationDataHandle Log(const ComputationDataHandle& operand);

  // Enqueues a sign instruction onto the computation.
  ComputationDataHandle Sign(const ComputationDataHandle& operand);

  // Enqueues a cosine instruction onto the computation.
  ComputationDataHandle Cos(const ComputationDataHandle& operand);

  // Enqueues a sine instruction onto the computation.
  ComputationDataHandle Sin(const ComputationDataHandle& operand);

  // Enqueues a tanh instruction onto the computation.
  ComputationDataHandle Tanh(const ComputationDataHandle& operand);

  // Enqueues a real-part instruction onto the computation.
  ComputationDataHandle Real(const ComputationDataHandle& operand);

  // Enqueues an imaginary-part instruction onto the computation.
  ComputationDataHandle Imag(const ComputationDataHandle& operand);

  // Enqueues a float32 sqrt instruction onto the computation.
  // (float32 is specified as there is an implicit float32 0.5f constant
  // exponent).
  ComputationDataHandle SqrtF32(const ComputationDataHandle& operand);

  // Enqueues a float32 square instruction onto the computation.
  // (float32 is specified as there is an implicit float32 2.0f constant
  // exponent).
  ComputationDataHandle SquareF32(const ComputationDataHandle& operand);

  // Enqueues a lhs^rhs computation onto the computation.
  ComputationDataHandle Pow(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues an operator that tests if the operand's values are finite, i.e.,
  // not Inf or NaN. Defined only for floating-point types. Returns an array of
  // booleans with the same shape where entries are true iff the corresponding
  // entry was NaN.
  ComputationDataHandle IsFinite(const ComputationDataHandle& operand);

  // Enqueues a convert instruction onto the computation that changes the
  // element type of the operand array to primitive_type.
  ComputationDataHandle ConvertElementType(const ComputationDataHandle& operand,
                                           PrimitiveType new_element_type);

  // Enqueues a no-op instruction onto the computation that changes
  // the element type of the operand array to primitive_type. The
  // bit-widths of the source and destination element types must be
  // identical.
  ComputationDataHandle BitcastConvertType(const ComputationDataHandle& operand,
                                           PrimitiveType new_element_type);

  // Enqueues a float32 reciprocal instruction onto the computation.
  // (float32 is specified as there is an implicit float32 -1.0f constant
  // exponent).
  //
  // TODO(b/34468990) axe F32 suffix, can be determined by reflecting on the
  // shape of the operand.
  ComputationDataHandle ReciprocalF32(const ComputationDataHandle& operand);

  // Enqueues a negate instruction onto the computation.
  ComputationDataHandle Neg(const ComputationDataHandle& operand);

  // Enqueues a transpose instruction onto the computation.
  ComputationDataHandle Transpose(
      const ComputationDataHandle& operand,
      tensorflow::gtl::ArraySlice<int64> permutation);

  // Enqueues a reverse instruction onto the computation. The order of the
  // elements in the given dimensions is reversed (i.e., the element at index i
  // is moved to index dimension_size - 1 - i).
  ComputationDataHandle Rev(const ComputationDataHandle& operand,
                            tensorflow::gtl::ArraySlice<int64> dimensions);

  // Enqueues a sort (as increasing order) instruction onto the computation.
  ComputationDataHandle Sort(const ComputationDataHandle& operand);

  // Enqueues a clamp instruction onto the computation.
  ComputationDataHandle Clamp(const ComputationDataHandle& min,
                              const ComputationDataHandle& operand,
                              const ComputationDataHandle& max);

  // Enqueues a map instruction onto the computation.
  ComputationDataHandle Map(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
      const Computation& computation,
      tensorflow::gtl::ArraySlice<int64> dimensions,
      tensorflow::gtl::ArraySlice<ComputationDataHandle> static_operands = {});

  // Enqueues a N(mu, sigma) random number generation instruction onto the
  // computation.
  ComputationDataHandle RngNormal(const ComputationDataHandle& mu,
                                  const ComputationDataHandle& sigma,
                                  const Shape& shape);

  // Enqueues a U(a, b) random number generation instruction onto the
  // computation. Returns values in the semi-open interval [a, b).
  ComputationDataHandle RngUniform(const ComputationDataHandle& a,
                                   const ComputationDataHandle& b,
                                   const Shape& shape);

  // Enqueues a while node onto the computation.
  ComputationDataHandle While(const Computation& condition,
                              const Computation& body,
                              const ComputationDataHandle& init);

  // Enqueues a conditional node onto the computation.
  ComputationDataHandle Conditional(const ComputationDataHandle& predicate,
                                    const ComputationDataHandle& true_operand,
                                    const Computation& true_computation,
                                    const ComputationDataHandle& false_operand,
                                    const Computation& false_computation);

  // Enqueues a ReducePrecision node onto the computation.
  ComputationDataHandle ReducePrecision(const ComputationDataHandle& operand,
                                        const int exponent_bits,
                                        const int mantissa_bits);

  // Enqueues a Gather node onto the computation.
  ComputationDataHandle Gather(
      const ComputationDataHandle& input,
      const ComputationDataHandle& gather_indices,
      const GatherDimensionNumbers& dimension_numbers,
      tensorflow::gtl::ArraySlice<int64> window_bounds);

  // Enqueues a Send node onto the computation, to send the given operand to
  // a Recv instruction that shares the same channel handle.
  void Send(const ComputationDataHandle& operand, const ChannelHandle& handle);

  // Enqueues a Recv node onto the computation. The data comes from a Send
  // instruction that shares the same channel handle and its shape must
  // be the same as the given shape.
  ComputationDataHandle Recv(const Shape& shape, const ChannelHandle& handle);

  // Returns true if 'operand' is a compile-time constant. A compile-time
  // constant does not depend on parameters with index greater than or equal to
  // `num_parameters`, or on stateful operators such as `RngNormal` or `Infeed`.
  // Unlike `ComputeConstant`, `IsConstant` tests whether a computation is a
  // compile-time constant without evaluating the computation.
  StatusOr<bool> IsConstant(const ComputationDataHandle& operand,
                            int64 num_parameters = 0);

  // Normalizes operand across spatial and batch dimensions for each feature.
  //
  // Returns a tuple (normalized, batch_mean, batch_var) where `normalized`
  // is the normalized result and batch_mean and batch_var are the mean and
  // variance, respectively, across batch for the operand.
  ComputationDataHandle BatchNormTraining(const ComputationDataHandle& operand,
                                          const ComputationDataHandle& scale,
                                          const ComputationDataHandle& offset,
                                          float epsilon, int64 feature_index);

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
  ComputationDataHandle BatchNormInference(
      const ComputationDataHandle& operand, const ComputationDataHandle& scale,
      const ComputationDataHandle& offset, const ComputationDataHandle& mean,
      const ComputationDataHandle& variance, float epsilon,
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
  ComputationDataHandle BatchNormGrad(const ComputationDataHandle& operand,
                                      const ComputationDataHandle& scale,
                                      const ComputationDataHandle& batch_mean,
                                      const ComputationDataHandle& batch_var,
                                      const ComputationDataHandle& grad_output,
                                      float epsilon, int64 feature_index);

  // Computes the value of a constant indicated by a
  // ComputationDataHandle using a non-optimized interpreter on the host.
  //
  // The operand must be from the computation currently being built -
  // i.e., returned from this builder with no intervening call to
  // Build(). This happens to currently work regardless of that, but
  // that may stop working at any time.
  //
  // The operand must represent a constant value, which in this case
  // means that it must not statically depend on any parameter of the
  // computation that is being built other then the ones specified on the
  // parameter list. The parameters in the list will be indexed by their
  // parameter id property so the number of parameters specified should be at
  // least as many as the largest used parameter index.
  //
  // `IsConstant` can be used to test whether a computation is a compile-time
  // constant without evaluation it. `ComputeConstant` only succeeds for
  // computations where `IsConstant` returns true.
  //
  // This functionality can be useful when translating a computation
  // into XLA where something that looked dynamic is required by
  // XLA to be specified as a constant. E.g. the source
  // computation (outside of XLA) may include a dynamic
  // computation of the shape of something and ComputeConstant lets
  // you determine what the value of that computation is in the case
  // where the value can be determined at compile time.
  //
  // If output_layout is non-null, then the output of the computation
  // will be stored using that layout.
  StatusOr<std::unique_ptr<Literal>> ComputeConstant(
      const ComputationDataHandle& operand,
      const Layout* output_layout = nullptr,
      tensorflow::gtl::ArraySlice<Literal> parameters = {});

  // Returns a new ComputationBuilder whose resultant Computation is used only
  // by this ComputationBuilder. The sub-ComputationBuilder has the same
  // die_immediately_on_error behavior as the parent.
  std::unique_ptr<ComputationBuilder> CreateSubBuilder(
      const string& computation_name);

  // Modifies the computation being built so that executions of it
  // will return the value associated with operand, rather than the
  // last expression enqueued on the ComputationBuilder. Any subsequent
  // operations added to the ComputationBuilder will not have any effect unless
  // SetReturnValue is called again.
  Status SetReturnValue(const ComputationDataHandle& operand);

  // Builds the computation with the requested operations, or returns a non-ok
  // status.
  StatusOr<Computation> Build();

  // Builds the computation with the requested operations, or notes an error in
  // the parent ComputationBuilder and returns an empty computation if building
  // failed. This function is intended to be used where the returned
  // Computation is only used by the parent ComputationBuilder and hence further
  // operation on the returned Computation will simply be error'ed out if an
  // error occurred while building this computation. If the built computation is
  // to be used by a ComputationBuilder other than the parent ComputationBuilder
  // then Build() should be used instead.
  Computation BuildAndNoteError();

  // Returns the first error that was encountered while building the
  // computation. When an error is encountered, by default we return a vacuous
  // ComputationDataHandle and inform the user of the error that occurred while
  // building the computation when they make a final call to Build().
  //
  // See also set_die_immediately_on_error().
  Status first_error() const { return first_error_; }

 private:
  // Limited checking of convolution parameters. Returns false on
  // error.
  bool VerifyConvolution(const Shape& lhs_shape, const Shape& rhs_shape,
                         const ConvolutionDimensionNumbers& dimension_numbers);

  // The parent ComputationBuilder of a sub-ComputationBuilder. The
  // parent_builder_ will be the nullptr if not a sub-ComputationBuilder.
  ComputationBuilder* parent_builder_{nullptr};

  // Helper function for creating a Window proto from user-supplied
  // data. Returns true if the user-supplied data was valid.
  bool MakeWindow(tensorflow::gtl::ArraySlice<int64> window_dimensions,
                  tensorflow::gtl::ArraySlice<int64> window_strides,
                  tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
                  tensorflow::gtl::ArraySlice<int64> lhs_dilation,
                  tensorflow::gtl::ArraySlice<int64> rhs_dilation,
                  Window* window);

  // Internal helper method that does the building for an arbitrary unary op.
  ComputationDataHandle UnaryOp(UnaryOperation unop,
                                const ComputationDataHandle& operand);

  // Internal helper method that does the building for an arbitrary binary op.
  // broadcast_dimensions specifies which dimensions to use for broadcasting
  // when the operation is between tensors of different ranks.
  ComputationDataHandle BinaryOp(
      BinaryOperation binop, const ComputationDataHandle& lhs,
      const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  // Internal helper method that does the building for an arbitrary ternary op.
  ComputationDataHandle TernaryOp(TernaryOperation triop,
                                  const ComputationDataHandle& lhs,
                                  const ComputationDataHandle& rhs,
                                  const ComputationDataHandle& ehs);

  // Internal helper method that does the building for a random number generator
  // of a given distribution with an explicitly specified shape.
  ComputationDataHandle RngOp(
      RandomDistribution distribution,
      tensorflow::gtl::ArraySlice<ComputationDataHandle> parameters,
      const Shape& shape);

  // Populates computation_ with a valid object or returns a failing status.
  // This is used before any given operation is enqueued.
  Status PrepareComputation();

  // Notes that the error occurred by:
  // * storing it internally and capturing a backtrace if it's the first error
  //   (this deferred value will be produced on the call to Build())
  // * dying if die_immediately_on_error_ is true
  void NoteError(const Status& error);

  // Helper function that runs the given op_request, filling in op_response.
  // Before the op is run, PrepareComputation is called, and common fields in
  // the op_request are filled in.
  Status RunOp(OpRequest* op_request, OpResponse* op_response);

  // Helper function that calls RunOp and calls NoteError on failures.
  void RunOpAndNoteError(OpRequest* op_request);

  // Helper function that calls RunOp and either returns the output computation
  // data handle (on success) or a vacuous computation data handle (on failure).
  ComputationDataHandle RunOpAndParseResponse(OpRequest* op_request);

  // Helper function that implements GetShape without noting errors. This makes
  // it easier to ensure the real GetShape will note errors on every error path.
  StatusOr<std::unique_ptr<Shape>> GetShapeWithoutNoteError(
      const ComputationDataHandle& operand);

  string name_;  // Name to use for the built computation.

  // The first error encountered while building the computation.
  // This is OK until the first error is encountered.
  Status first_error_;

  // The saved stack trace from the point at which the first error occurred.
  tensorflow::SavedStackTrace first_error_backtrace_;

  // The computation that operations are enqueued onto.
  Computation computation_;

  // The client that the computation is created in. Not owned.
  Client* client_;

  // Mode bit that indicates whether to die when a first error is encountered.
  bool die_immediately_on_error_ = false;

  // The metadata to attach to each op. This is structured as a "modal"-like
  // operation, in order to simplify client code (and not sprinkle this metadata
  // throughout the TensorFlow op kernel implementations).
  OpMetadata metadata_;

  // Sharding for this operator. This is structured as a "model"-like operation,
  // in order to simplify client code, similar to metadata_.
  tensorflow::gtl::optional<OpSharding> sharding_;

  TF_DISALLOW_COPY_AND_ASSIGN(ComputationBuilder);
};

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR0(NativeT value) {
  return ConstantLiteral(*Literal::CreateR0<NativeT>(value));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR1(
    tensorflow::gtl::ArraySlice<NativeT> values) {
  return ConstantLiteral(*Literal::CreateR1<NativeT>(values));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR1(int64 length,
                                                     NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {length}));
  literal.PopulateWithValue(value);
  return ConstantLiteral(literal);
}

inline ComputationDataHandle ComputationBuilder::ConstantR1(
    const tensorflow::core::Bitmap& values) {
  return ConstantLiteral(*Literal::CreateR1(values));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return ConstantLiteral(*Literal::CreateR2<NativeT>(values));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantFromArrayWithLayout(
    const Array<NativeT>& values, const Layout& layout) {
  return ConstantLiteral(
      *Literal::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantFromArray(
    const Array<NativeT>& values) {
  return ConstantLiteral(*Literal::CreateFromArray<NativeT>(values));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout) {
  return ConstantLiteral(
      *Literal::CreateFromArrayWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR2FromArray2D(
    const Array2D<NativeT>& values) {
  return ConstantLiteral(*Literal::CreateR2FromArray2D<NativeT>(values));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout) {
  return ConstantLiteral(
      *Literal::CreateR3FromArray3DWithLayout<NativeT>(values, layout));
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR3FromArray3D(
    const Array3D<NativeT>& values) {
  return ConstantFromArray(values);
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout) {
  return ConstantFromArrayWithLayout(values, layout);
}

template <typename NativeT>
ComputationDataHandle ComputationBuilder::ConstantR4FromArray4D(
    const Array4D<NativeT>& values) {
  return ConstantFromArray(values);
}

// RAII-style object: sets the current sharding assignment in builder on
// construction, and sets back to the previous assignment on destruction.
class ScopedShardingAssignment {
 public:
  ScopedShardingAssignment(xla::ComputationBuilder* builder,
                           tensorflow::gtl::optional<OpSharding> sharding)
      : builder_(builder), prev_sharding_(builder->sharding()) {
    SetSharding(sharding);
  }

  ~ScopedShardingAssignment() { SetSharding(prev_sharding_); }

 private:
  void SetSharding(const tensorflow::gtl::optional<OpSharding>& sharding) {
    if (sharding.has_value()) {
      builder_->SetSharding(sharding.value());
    } else {
      builder_->ClearSharding();
    }
  }

  xla::ComputationBuilder* const builder_;
  tensorflow::gtl::optional<OpSharding> prev_sharding_;

  TF_DISALLOW_COPY_AND_ASSIGN(ScopedShardingAssignment);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_COMPUTATION_BUILDER_H_
