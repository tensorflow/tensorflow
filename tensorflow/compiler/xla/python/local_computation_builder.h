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
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
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
  static StatusOr<LocalShapedBuffer*> FromLiteral(
      const Literal& argument,
      const tensorflow::gtl::optional<Shape>& shape_with_layout);

  LocalShapedBuffer(ScopedShapedBuffer shaped_buffer);
  const ScopedShapedBuffer* shaped_buffer() const;

  StatusOr<std::unique_ptr<Literal> > ToLiteral() const;

  // Transfers ownership of the encapsulated ShapedBuffer to the caller,
  // analogous to std::unique_ptr::release().
  ShapedBuffer Release();

 private:
  ScopedShapedBuffer shaped_buffer_;
};

// Result of a tuple destructuring operation on a LocalShapedBuffer -- this
// appears to be a simpler mechanism for the time being than an alternative like
// using SWIG to transform std::vectors into Python lists of SWIG objects
// directly.
class LocalShapedBufferTuple {
 public:
  // Note: any LocalShapedBuffer elements that are not Release()'d will be
  // deallocated in the destructor.
  explicit LocalShapedBufferTuple(std::vector<LocalShapedBuffer*> elements);

  ~LocalShapedBufferTuple();

  // Releases the ith element to the caller. Further attempts to release the ith
  // element will return an invalid argument error.
  StatusOr<LocalShapedBuffer*> Release(int i);

  // Returns the number of elements in the destructured tuple.
  int size() const;

 private:
  std::vector<LocalShapedBuffer*> elements_;
};

// Destructures a tuple-valued LocalShapedBuffer into its constitutent elements
// in LocalShapedBufferTuple form.
StatusOr<LocalShapedBufferTuple*> DestructureLocalShapedBufferTuple(
    LocalShapedBuffer* local_shaped_buffer);

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

// Wraps a XlaComputation produced by a LocalComputationBuilder. The
// Compile method compiles the computation to a (local) executable via
// the client library's local client. This class is intended to be
// made available to Python via SWIG.
class LocalComputation {
 public:
  LocalComputation(XlaComputation computation);

  StatusOr<CompiledLocalComputation*> Compile(
      const std::vector<Shape>& argument_shapes,
      const ExecutableBuildOptions* build_options);

  const XlaComputation& computation() const;

  // Returns the HloModuleProto contained in the XlaComputation in the
  // serialized binary format. Logs an internal error and returns an empty
  // string on failure.
  string GetSerializedProto() const;

  // Returns the return-value shape for this computation.
  StatusOr<Shape> GetReturnValueShape() const;

 private:
  XlaComputation computation_;
};

// Wraps a XlaOp produced by a LocalComputationBuilder. This class is intended
// to be made available to Python via SWIG.
class LocalOp {
 public:
  LocalOp(const XlaOp& op);

  const XlaOp& op() const;

 private:
  XlaOp op_;
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

  LocalOp Parameter(int64 parameter_number, const Shape& shape,
                    const string& name);

  StatusOr<Shape> GetShape(const LocalOp& operand);

  // Returns the shape of the current return value for the computation.
  StatusOr<Shape> GetReturnValueShape();

  LocalOp Infeed(const Shape& shape);

  void Outfeed(const LocalOp& operand, const Shape& shape,
               const string& outfeed_config);

  LocalOp ConstantLiteral(const Literal& literal);

  LocalOp Broadcast(const LocalOp& operand,
                    tensorflow::gtl::ArraySlice<int64> broadcast_sizes);

  LocalOp Pad(const LocalOp& operand, const LocalOp& padding_value,
              const PaddingConfig& padding_config);

  LocalOp Reshape(const LocalOp& operand,
                  tensorflow::gtl::ArraySlice<int64> dimensions,
                  tensorflow::gtl::ArraySlice<int64> new_sizes);

  LocalOp Collapse(const LocalOp& operand,
                   tensorflow::gtl::ArraySlice<int64> dimensions);

  LocalOp CrossReplicaSum(const LocalOp& operand);

  LocalOp Slice(const LocalOp& operand,
                tensorflow::gtl::ArraySlice<int64> start_indices,
                tensorflow::gtl::ArraySlice<int64> limit_indices,
                tensorflow::gtl::ArraySlice<int64> strides);

  LocalOp SliceInDim(const LocalOp& operand, int64 start_index,
                     int64 limit_index, int64 stride, int64 dimno);

  LocalOp DynamicSlice(const LocalOp& operand, const LocalOp& start_indices,
                       tensorflow::gtl::ArraySlice<int64> slice_sizes);

  LocalOp DynamicUpdateSlice(const LocalOp& operand, const LocalOp& update,
                             const LocalOp& start_indices);

  LocalOp ConcatInDim(tensorflow::gtl::ArraySlice<LocalOp> operands,
                      int64 dimension);

  LocalOp SelectAndScatterWithGeneralPadding(
      const LocalOp& operand, const LocalComputation& select,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64> > padding,
      const LocalOp& source, const LocalOp& init_value,
      const LocalComputation& scatter);

  LocalOp Tuple(tensorflow::gtl::ArraySlice<LocalOp> elements);

  LocalOp GetTupleElement(const LocalOp& tuple_data, int64 index);

  LocalOp Dot(const LocalOp& lhs, const LocalOp& rhs);

  LocalOp DotGeneral(const LocalOp& lhs, const LocalOp& rhs,
                     const DotDimensionNumbers& dimension_numbers);

  LocalOp ConvGeneralDilated(
      const LocalOp& lhs, const LocalOp& rhs,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64> > padding,
      tensorflow::gtl::ArraySlice<int64> lhs_dilation,
      tensorflow::gtl::ArraySlice<int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers);

  LocalOp ConvertElementType(const LocalOp& operand,
                             PrimitiveType new_element_type);

  LocalOp BitcastConvertType(const LocalOp& operand,
                             PrimitiveType new_element_type);

  LocalOp Call(const LocalComputation& local_computation,
               tensorflow::gtl::ArraySlice<LocalOp> operands);

  LocalOp Transpose(const LocalOp& operand,
                    tensorflow::gtl::ArraySlice<int64> permutation);

  LocalOp Rev(const LocalOp& operand,
              tensorflow::gtl::ArraySlice<int64> dimensions);

  LocalOp Map(tensorflow::gtl::ArraySlice<LocalOp> operands,
              const LocalComputation& local_computation,
              tensorflow::gtl::ArraySlice<int64> dimensions);

  LocalOp Reduce(const LocalOp& operand, const LocalOp& init_value,
                 const LocalComputation& local_computation,
                 tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce);

  LocalOp ReduceWindowWithGeneralPadding(
      const LocalOp& operand, const LocalOp& init_value,
      const LocalComputation& local_computation,
      tensorflow::gtl::ArraySlice<int64> window_dimensions,
      tensorflow::gtl::ArraySlice<int64> window_strides,
      tensorflow::gtl::ArraySlice<std::pair<int64, int64> > padding);

  LocalOp RngNormal(const LocalOp& mu, const LocalOp& sigma,
                    const Shape& shape);

  LocalOp RngUniform(const LocalOp& a, const LocalOp& b, const Shape& shape);

  LocalOp While(const LocalComputation& condition, const LocalComputation& body,
                const LocalOp& init);

  LocalOp Conditional(const LocalOp& predicate, const LocalOp& true_operand,
                      const LocalComputation& true_computation,
                      const LocalOp& false_operand,
                      const LocalComputation& false_computation);

  StatusOr<bool> IsConstant(const LocalOp& operand);

  StatusOr<LocalComputation*> BuildConstantSubGraph(const LocalOp& operand);

#define _FORWARD(method_name, return_sig, args_sig) \
  return_sig method_name args_sig;

#define _FORWARD_UNOP(method_name) \
  _FORWARD(method_name, LocalOp, (const LocalOp& operand))

#define _FORWARD_BINOP(method_name)                 \
  _FORWARD(method_name, LocalOp,                    \
           (const LocalOp& lhs, const LocalOp& rhs, \
            tensorflow::gtl::ArraySlice<int64> broadcast_dimensions))

#define _FORWARD_TRIOP(method_name) \
  _FORWARD(method_name, LocalOp,    \
           (const LocalOp& lhs, const LocalOp& rhs, const LocalOp& ehs))

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
  _FORWARD_BINOP(Xor)
  _FORWARD_BINOP(ShiftLeft)
  _FORWARD_BINOP(ShiftRightArithmetic)
  _FORWARD_BINOP(ShiftRightLogical)
  _FORWARD_BINOP(Atan2)
  _FORWARD_BINOP(Pow)
  _FORWARD_UNOP(Not)
  _FORWARD_UNOP(Abs)
  _FORWARD_UNOP(Exp)
  _FORWARD_UNOP(Expm1)
  _FORWARD_UNOP(Floor)
  _FORWARD_UNOP(Ceil)
  _FORWARD_UNOP(Round)
  _FORWARD_UNOP(Log)
  _FORWARD_UNOP(Log1p)
  _FORWARD_UNOP(Sign)
  _FORWARD_UNOP(Cos)
  _FORWARD_UNOP(Sin)
  _FORWARD_UNOP(Tanh)
  _FORWARD_UNOP(IsFinite)
  _FORWARD_UNOP(Neg)
  _FORWARD_UNOP(Sort)
  _FORWARD_UNOP(Sqrt)
  _FORWARD_UNOP(Rsqrt)
  _FORWARD_UNOP(Square)
  _FORWARD_UNOP(Reciprocal)
  _FORWARD_UNOP(Erfc)
  _FORWARD_UNOP(Erf)
  _FORWARD_UNOP(ErfInv)
  _FORWARD_UNOP(Lgamma)
  _FORWARD_UNOP(Digamma)
  _FORWARD_UNOP(Acos)
  _FORWARD_UNOP(Asin)
  _FORWARD_UNOP(Atan)
  _FORWARD_UNOP(Tan)
  _FORWARD_UNOP(Acosh)
  _FORWARD_UNOP(Asinh)
  _FORWARD_UNOP(Atanh)
  _FORWARD_UNOP(Cosh)
  _FORWARD_UNOP(Sinh)

#undef _FORWARD
#undef _FORWARD_UNOP
#undef _FORWARD_BINOP
#undef _FORWARD_TRIOP

 private:
  XlaBuilder builder_;
};

// Functions for freeing resources from the Python side.
void DeleteLocalShapedBuffer(LocalShapedBuffer* local_shaped_buffer);
void DeleteCompiledLocalComputation(CompiledLocalComputation* computation);
void DeleteLocalComputation(LocalComputation* computation);

}  // namespace swig
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_COMPUTATION_BUILDER_H_
