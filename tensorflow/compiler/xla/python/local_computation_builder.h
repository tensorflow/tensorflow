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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_COMPUTATION_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_COMPUTATION_BUILDER_H_

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

namespace swig {

// Initializes the number of replicas that XLA will be initialized with (when
// first obtaining a handle to the local XLA service). If this is called after
// the handle to the local XLA service has been established, then an error is
// returned.
Status InitializeReplicaCount(int replica_count);

// Returns the replica count that is currently set, regardless of whether the
// local XLA service has been instantiated yet or not.
int GetReplicaCount();

// Wraps the local client's infeed-transfer function.
//
// The default device ordinal (0) is used.
Status TransferToInfeedLocal(const Literal& literal);

// Transfers the given literal to the infeed of the given replica.
//
// The replica number is resolved to an appropriate device ordinal.
Status TransferToInfeedLocalReplica(const Literal& literal, int replica_number);

// Transfers a literal of the given shape from the outfeed of the given replica.
//
// The replica number is resolved to an appropriate device ordinal.
StatusOr<std::unique_ptr<Literal> > TransferFromOutfeedLocalReplica(
    const Shape& shape, int replica_number);

// Wraps a ScopedShapedBuffer produced by copying a literal "to
// device," i.e. copying a literal to a scoped buffer via the local
// client.
class LocalShapedBuffer {
 public:
  static LocalShapedBuffer* FromLiteral(
      const Literal& argument,
      const tensorflow::gtl::optional<Shape>& shape_with_layout);
  LocalShapedBuffer(std::unique_ptr<ScopedShapedBuffer> shaped_buffer);
  const std::unique_ptr<ScopedShapedBuffer>& shaped_buffer() const;
  std::unique_ptr<Literal> ToLiteral() const;

 private:
  std::unique_ptr<ScopedShapedBuffer> shaped_buffer_;
};

// Wraps a LocalExecutable produced by compiling a
// LocalComputation. The Execute method forwards to that of the
// underlying LocalExecutable, and additionally handles tranferring
// arguments and return values in and back out of the client library's
// local client. This class is intended to be made available to Python
// via SWIG.
class CompiledLocalComputation {
 public:
  CompiledLocalComputation(std::unique_ptr<LocalExecutable> executable);

  // Execute the computation with the given argument literals, and
  // with optionally-specified argument layouts. The literals will be
  // re-laid out according to the corresponding elements of
  // shapes_with_layout.
  StatusOr<std::unique_ptr<Literal> > Execute(
      const std::vector<Literal>& arguments,
      const std::vector<tensorflow::gtl::optional<Shape> >& shapes_with_layout);

  LocalShapedBuffer* ExecuteWithShapedBuffers(
      tensorflow::gtl::ArraySlice<LocalShapedBuffer*> argument_handles);

 private:
  std::unique_ptr<LocalExecutable> executable_;
};

// Wraps a Computation produced by a LocalComputationBuilder. The
// Compile method compiles the computation to a (local) executable via
// the client library's local client. This class is intended to be
// made available to Python via SWIG.
class LocalComputation {
 public:
  LocalComputation(Computation computation);

  StatusOr<CompiledLocalComputation*> Compile(
      const std::vector<Shape>& argument_shapes,
      const ExecutableBuildOptions* build_options);

  const Computation& computation() const;

  // Returns the return-value shape for this computation.
  StatusOr<Shape> GetReturnValueShape() const;

 private:
  Computation computation_;
};

// Wraps the ComputationBuilder API in order to:
// - Support consumption by SWIG in order to be made available to
//   Python.
// - Set up the underlying builder to use the client library's
//   LocalClient.
// - Wrap Computations in LocalComputations for Python access.
// - Correspondingly unwrap incoming LocalComputations.
class LocalComputationBuilder {
 public:
  LocalComputationBuilder(const string& computation_name);

  void SetOpMetadata(const OpMetadata& metadata);
  void ClearOpMetadata();

  // Returns an owned LocalComputation to the caller on success.
  StatusOr<LocalComputation*> Build();

  ComputationDataHandle Parameter(int64 parameter_number, const Shape& shape,
                                  const string& name);

  std::unique_ptr<Shape> GetShape(const ComputationDataHandle& operand);

  // Returns the shape of the current return value for the computation.
  StatusOr<Shape> GetReturnValueShape();

  ComputationDataHandle Infeed(const Shape& shape);

  void Outfeed(const ComputationDataHandle& operand, const Shape& shape,
               const string& outfeed_config);

  ComputationDataHandle ConstantLiteral(const Literal& literal);

  ComputationDataHandle Broadcast(
      const ComputationDataHandle& operand,
      tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

  ComputationDataHandle Pad(const ComputationDataHandle& operand,
                            const ComputationDataHandle& padding_value,
                            const PaddingConfig& padding_config);

  ComputationDataHandle Reshape(const ComputationDataHandle& operand,
                                tensorflow::gtl::ArraySlice<int64> dimensions,
                                tensorflow::gtl::ArraySlice<int64> new_sizes);

  ComputationDataHandle Collapse(const ComputationDataHandle& operand,
                                 tensorflow::gtl::ArraySlice<int64> dimensions);

  ComputationDataHandle CrossReplicaSum(const ComputationDataHandle& operand);

  ComputationDataHandle Slice(const ComputationDataHandle& operand,
                              tensorflow::gtl::ArraySlice<int64> start_indices,
                              tensorflow::gtl::ArraySlice<int64> limit_indices,
                              tensorflow::gtl::ArraySlice<int64> strides);

  ComputationDataHandle SliceInDim(const ComputationDataHandle& operand,
                                   int64 start_index, int64 limit_index,
                                   int64 stride, int64 dimno);

  ComputationDataHandle DynamicSlice(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& start_indices,
      tensorflow::gtl::ArraySlice<int64> slice_sizes);

  ComputationDataHandle DynamicUpdateSlice(
      const ComputationDataHandle& operand, const ComputationDataHandle& update,
      const ComputationDataHandle& start_indices);

  ComputationDataHandle ConcatInDim(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
      int64 dimension);

  ComputationDataHandle SelectAndScatterWithGeneralPadding(
      const ComputationDataHandle& operand, const LocalComputation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64> > padding,
      const ComputationDataHandle& source,
      const ComputationDataHandle& init_value, const LocalComputation& scatter);

  ComputationDataHandle Tuple(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> elements);

  ComputationDataHandle GetTupleElement(const ComputationDataHandle& tuple_data,
                                        int64 index);

  ComputationDataHandle Dot(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs);

  ComputationDataHandle DotGeneral(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      const DotDimensionNumbers& dimension_numbers);

  ComputationDataHandle ConvGeneralDilated(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64> > padding,
      tensorflow::gtl::ArraySlice<int64> lhs_dilation,
      tensorflow::gtl::ArraySlice<int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers);

  ComputationDataHandle ConvertElementType(const ComputationDataHandle& operand,
                                           PrimitiveType new_element_type);

  ComputationDataHandle Call(
      const LocalComputation& local_computation,
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands);

  ComputationDataHandle Transpose(
      const ComputationDataHandle& operand,
      tensorflow::gtl::ArraySlice<int64> permutation);

  ComputationDataHandle Rev(const ComputationDataHandle& operand,
                            tensorflow::gtl::ArraySlice<int64> dimensions);

  ComputationDataHandle Map(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
      const LocalComputation& local_computation,
      tensorflow::gtl::ArraySlice<int64> dimensions,
      tensorflow::gtl::ArraySlice<ComputationDataHandle> static_operands);

  ComputationDataHandle Reduce(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& init_value,
      const LocalComputation& local_computation,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce);

  ComputationDataHandle ReduceWindowWithGeneralPadding(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& init_value,
      const LocalComputation& local_computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64> > padding);

  ComputationDataHandle RngNormal(const ComputationDataHandle& mu,
                                  const ComputationDataHandle& sigma,
                                  const Shape& shape);

  ComputationDataHandle RngUniform(const ComputationDataHandle& a,
                                   const ComputationDataHandle& b,
                                   const Shape& shape);

  ComputationDataHandle While(const LocalComputation& condition,
                              const LocalComputation& body,
                              const ComputationDataHandle& init);

  ComputationDataHandle Conditional(const ComputationDataHandle& predicate,
                                    const ComputationDataHandle& true_operand,
                                    const LocalComputation& true_computation,
                                    const ComputationDataHandle& false_operand,
                                    const LocalComputation& false_computation);

  StatusOr<bool> IsConstant(const ComputationDataHandle& operand,
                            int64 num_parameters);

  StatusOr<std::unique_ptr<Literal> > ComputeConstant(
      const ComputationDataHandle& operand, const Layout* output_layout,
      tensorflow::gtl::ArraySlice<Literal> parameters);

#define _FORWARD(method_name, return_sig, args_sig) \
  return_sig method_name args_sig;

#define _FORWARD_UNOP(method_name)             \
  _FORWARD(method_name, ComputationDataHandle, \
           (const ComputationDataHandle& operand))

#define _FORWARD_BINOP(method_name)                                        \
  _FORWARD(                                                                \
      method_name, ComputationDataHandle,                                  \
      (const ComputationDataHandle& lhs, const ComputationDataHandle& rhs, \
       tensorflow::gtl::ArraySlice<int64> broadcast_dimensions))

#define _FORWARD_TRIOP(method_name)                                        \
  _FORWARD(                                                                \
      method_name, ComputationDataHandle,                                  \
      (const ComputationDataHandle& lhs, const ComputationDataHandle& rhs, \
       const ComputationDataHandle& ehs))

  _FORWARD_TRIOP(Select)
  _FORWARD_TRIOP(Clamp)
  _FORWARD_BINOP(Eq)
  _FORWARD_BINOP(Ne)
  _FORWARD_BINOP(Ge)
  _FORWARD_BINOP(Gt)
  _FORWARD_BINOP(Lt)
  _FORWARD_BINOP(Le)
  _FORWARD_BINOP(Add)
  _FORWARD_BINOP(Sub)
  _FORWARD_BINOP(Mul)
  _FORWARD_BINOP(Div)
  _FORWARD_BINOP(Rem)
  _FORWARD_BINOP(Max)
  _FORWARD_BINOP(Min)
  _FORWARD_BINOP(And)
  _FORWARD_BINOP(Or)
  _FORWARD_UNOP(Not)
  _FORWARD_UNOP(Abs)
  _FORWARD_UNOP(Exp)
  _FORWARD_UNOP(Floor)
  _FORWARD_UNOP(Ceil)
  _FORWARD_UNOP(Round)
  _FORWARD_UNOP(Log)
  _FORWARD_UNOP(Sign)
  _FORWARD_UNOP(Cos)
  _FORWARD_UNOP(Sin)
  _FORWARD_UNOP(Tanh)
  _FORWARD_UNOP(SqrtF32)
  _FORWARD_UNOP(SquareF32)
  _FORWARD_BINOP(Pow)
  _FORWARD_UNOP(IsFinite)
  _FORWARD_UNOP(ReciprocalF32)
  _FORWARD_UNOP(Neg)
  _FORWARD_UNOP(Sort)

#undef _FORWARD
#undef _FORWARD_UNOP
#undef _FORWARD_BINOP
#undef _FORWARD_TRIOP

 private:
  ComputationBuilder builder_;
};

// Functions for freeing resources from the Python side.
void DeleteLocalShapedBuffer(LocalShapedBuffer* local_shaped_buffer);
void DeleteCompiledLocalComputation(CompiledLocalComputation* computation);
void DeleteLocalComputation(LocalComputation* computation);

}  // namespace swig

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_COMPUTATION_BUILDER_H_
