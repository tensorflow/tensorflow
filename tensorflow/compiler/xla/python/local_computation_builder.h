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
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

namespace swig {

// Wraps a LocalExecutable produced by compiling a
// LocalComputation. The Execute method forwards to that of the
// underlying LocalExecutable, and additionally handles tranferring
// arguments and return values in and back out of the client library's
// local client. This class is intended to be made available to Python
// via SWIG.
class CompiledLocalComputation {
 public:
  CompiledLocalComputation(std::unique_ptr<LocalExecutable> executable);
  std::unique_ptr<Literal> Execute(const std::vector<Literal>& arguments);

 private:
  std::unique_ptr<LocalExecutable> executable_;
};

// Wraps a Computation produced by a LocalComputationBuilder. The
// Compile method compiles the computation to a (local) executable via
// the client library's local client. This class is intended to be
// made available to Python via SWIG.
class LocalComputation {
 public:
  LocalComputation(std::unique_ptr<Computation> computation);
  CompiledLocalComputation* Compile(const std::vector<Shape>& argument_shapes);
  const Computation& computation() const;

 private:
  std::unique_ptr<Computation> computation_;
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

  LocalComputation* Build();

  ComputationDataHandle Parameter(int64 parameter_number, const Shape& shape,
                                  const string& name);

  std::unique_ptr<Shape> GetShape(const ComputationDataHandle& operand);

  ComputationDataHandle ConstantLiteral(const Literal& literal);

  ComputationDataHandle Broadcast(
      const ComputationDataHandle& operand,
      tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

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

  ComputationDataHandle Select(const ComputationDataHandle& pred,
                               const ComputationDataHandle& on_true,
                               const ComputationDataHandle& on_false);

  ComputationDataHandle Tuple(
      tensorflow::gtl::ArraySlice<ComputationDataHandle> elements);

  ComputationDataHandle GetTupleElement(const ComputationDataHandle& tuple_data,
                                        int64 index);

  ComputationDataHandle Dot(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs);

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

  ComputationDataHandle While(const LocalComputation& condition,
                              const LocalComputation& body,
                              const ComputationDataHandle& init);

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

 private:
  ComputationBuilder builder_;
};

static void DeleteLocalComputation(LocalComputation* computation) {
  delete computation;
}

static void DeleteCompiledLocalComputation(
    CompiledLocalComputation* computation) {
  delete computation;
}

}  // namespace swig

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_COMPUTATION_BUILDER_H_
