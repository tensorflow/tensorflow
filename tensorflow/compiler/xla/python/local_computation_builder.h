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

#include <string>
#include <vector>

#include <Python.h>

#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace swig {

// Initializes the number of replicas that XLA will be initialized with (when
// first obtaining a handle to the local XLA service). If this is called after
// the handle to the local XLA service has been established, then an error is
// returned.
Status InitializeReplicaCount(int replica_count);

// Initializes the platform name that XLA will be initialized with (when
// first obtaining a handle to the local XLA service). If this is called after
// the handle to the local XLA service has been established, then an error is
// returned.
Status InitializePlatformName(const string& platform_name);

// Returns the replica count that is currently set, regardless of whether the
// local XLA service has been instantiated yet or not.
int GetReplicaCount();

// Registers a 'fn_capsule' as a CPU custom call target.
// 'fn_capsule' is a void* pointer encapsulated in a PyCapsule object, with name
// "xla._CPU_CUSTOM_CALL_TARGET".
Status RegisterCpuCustomCallTarget(const string& name, PyObject* fn_capsule);

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
StatusOr<Literal> TransferFromOutfeedLocalReplica(const Shape& shape,
                                                  int replica_number);

// Represents a reference to literals that live in a device-allocated buffer via
// XLA. Specifically, wraps a ScopedShapedBuffer produced by transferring a
// literal to device via the local client.
class LocalShapedBuffer {
 public:
  static StatusOr<LocalShapedBuffer*> FromLiteral(
      const Literal& argument, const absl::optional<Shape>& shape_with_layout,
      int replica_number);

  LocalShapedBuffer(ScopedShapedBuffer shaped_buffer);
  StatusOr<Literal> ToLiteral() const;
  const Shape& shape() const;
  const ScopedShapedBuffer* shaped_buffer() const;

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
  int64 size() const;

 private:
  std::vector<LocalShapedBuffer*> elements_;
};

// Destructures a tuple-valued LocalShapedBuffer into its constitutent elements
// in LocalShapedBufferTuple form.
StatusOr<LocalShapedBufferTuple*> DestructureLocalShapedBufferTuple(
    LocalShapedBuffer* local_shaped_buffer);

// Represents a reference to literals that live in a device-allocated buffer via
// XRT. Specifically, wraps an int64 handle produced by running the allocation
// graph, and an XLA shape to track the referent's shape.
class XrtAllocation {
 public:
  // Accepts a `session_target` argument, used in constructing the
  // `tensorflow::ClientSession` instance in which allocation and deallocation
  // graphs are run.
  static StatusOr<XrtAllocation*> FromLiteral(const Literal& argument,
                                              const string& session_target);

  XrtAllocation(int64 handle, Shape shape, const string& session_target);
  ~XrtAllocation();
  StatusOr<Literal> ToLiteral() const;
  const Shape& shape() const;
  const int64 handle() const;

 private:
  const int64 handle_;
  const Shape shape_;
  const string session_target_;
};

// Result of a tuple destructuring operation on an XrtAllocation.
class XrtAllocationTuple {
 public:
  // Note: any XrtAllocation elements that are not Release()'d will be
  // deallocated in the destructor.
  explicit XrtAllocationTuple(std::vector<XrtAllocation*> elements);

  ~XrtAllocationTuple();

  // Releases the ith element to the caller. Further attempts to release the ith
  // element will return an invalid argument error.
  StatusOr<XrtAllocation*> Release(int i);

  // Returns the number of elements in the destructured tuple.
  int64 size() const;

 private:
  std::vector<XrtAllocation*> elements_;
};

// Destructures a tuple-valued XrtAllocation into its constitutent elements
// in XrtAllocationTuple form.
//
// Accepts a `session_target` argument, used in constructing the
// `tensorflow::ClientSession` instance in which the sub-tupling graph is run,
// and passed along in constructing each constituent XrtAllocation.
StatusOr<XrtAllocationTuple*> DestructureXrtAllocationTuple(
    XrtAllocation* allocation, const string& session_target);

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Specifically, wraps an XLA LocalExecutable.
class CompiledLocalComputation {
 public:
  CompiledLocalComputation(std::unique_ptr<LocalExecutable> executable);

  int num_replicas() const {
    return executable_->build_options().num_replicas();
  }

  StatusOr<LocalShapedBuffer*> Execute(
      absl::Span<LocalShapedBuffer* const> argument_handles);

  // Execute on many replicas. Takes a sequence of argument lists (one argument
  // list per replica) and returns a tuple of results (one result per replica).
  // The number of argument lists must be equal to the replica count.
  StatusOr<LocalShapedBufferTuple*> ExecutePerReplica(
      absl::Span<const std::vector<LocalShapedBuffer*> > argument_handles);

 private:
  std::unique_ptr<LocalExecutable> executable_;
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Specifically, wraps an XRT computation handle.
class CompiledXrtComputation {
 public:
  // Accepts a `session_target` argument, used in constructing the
  // `tensorflow::ClientSession` instance in which the execution graph is run.
  CompiledXrtComputation(const ProgramShape& program_shape, int64 handle,
                         const string& session_target);
  ~CompiledXrtComputation();

  StatusOr<XrtAllocation*> Execute(
      absl::Span<XrtAllocation* const> argument_handles);

  const ProgramShape& program_shape() const;
  int64 handle() const;

 private:
  const ProgramShape program_shape_;
  const int64 handle_;
  const string session_target_;
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

  // Accepts a `session_target` argument, used in constructing the
  // `tensorflow::ClientSession` instance in which the compilation graph is run.
  StatusOr<CompiledXrtComputation*> CompileForXrt(
      const std::vector<Shape>& argument_shapes, const string& session_target);

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

  // Returns an owned LocalComputation to the caller on success with given root.
  StatusOr<LocalComputation*> BuildWithRoot(const LocalOp& root);

  LocalOp Parameter(int64 parameter_number, const Shape& shape,
                    const string& name);

  StatusOr<Shape> GetShape(const LocalOp& operand);

  // Returns the shape of the current return value for the computation.
  StatusOr<Shape> GetReturnValueShape();

  LocalOp Infeed(const Shape& shape);

  void Outfeed(const LocalOp& operand, const Shape& shape,
               const string& outfeed_config);

  LocalOp ConstantLiteral(const Literal& literal);

  LocalOp Iota(PrimitiveType element_type, int64 size);

  LocalOp BroadcastedIota(const Shape& shape, int64 dimension);

  LocalOp Broadcast(const LocalOp& operand,
                    absl::Span<const int64> broadcast_sizes);

  LocalOp BroadcastInDim(const LocalOp& operand,
                         absl::Span<const int64> out_dim_sizes,
                         absl::Span<const int64> broadcast_dimensions);

  LocalOp Pad(const LocalOp& operand, const LocalOp& padding_value,
              const PaddingConfig& padding_config);

  LocalOp Reshape(const LocalOp& operand, absl::Span<const int64> dimensions,
                  absl::Span<const int64> new_sizes);

  LocalOp Collapse(const LocalOp& operand, absl::Span<const int64> dimensions);

  LocalOp AllToAll(const LocalOp& operand, int64 split_dimension,
                   int64 concat_dimension, int64 split_count,
                   absl::Span<const ReplicaGroup> replica_groups);

  LocalOp CrossReplicaSum(const LocalOp& operand,
                          absl::Span<const ReplicaGroup> replica_groups);

  LocalOp Slice(const LocalOp& operand, absl::Span<const int64> start_indices,
                absl::Span<const int64> limit_indices,
                absl::Span<const int64> strides);

  LocalOp SliceInDim(const LocalOp& operand, int64 start_index,
                     int64 limit_index, int64 stride, int64 dimno);

  LocalOp DynamicSlice(const LocalOp& operand, const LocalOp& start_indices,
                       absl::Span<const int64> slice_sizes);

  LocalOp DynamicUpdateSlice(const LocalOp& operand, const LocalOp& update,
                             const LocalOp& start_indices);

  LocalOp ConcatInDim(absl::Span<const LocalOp> operands, int64 dimension);

  LocalOp SelectAndScatterWithGeneralPadding(
      const LocalOp& operand, const LocalComputation& select,
      absl::Span<const int64> window_dimensions,
      absl::Span<const int64> window_strides,
      absl::Span<const std::pair<int64, int64> > padding, const LocalOp& source,
      const LocalOp& init_value, const LocalComputation& scatter);

  LocalOp Tuple(absl::Span<const LocalOp> elements);

  LocalOp GetTupleElement(const LocalOp& tuple_data, int64 index);

  LocalOp Dot(const LocalOp& lhs, const LocalOp& rhs);

  LocalOp DotGeneral(const LocalOp& lhs, const LocalOp& rhs,
                     const DotDimensionNumbers& dimension_numbers);

  LocalOp ConvGeneralDilated(
      const LocalOp& lhs, const LocalOp& rhs,
      absl::Span<const int64> window_strides,
      absl::Span<const std::pair<int64, int64> > padding,
      absl::Span<const int64> lhs_dilation,
      absl::Span<const int64> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64 feature_group_count);

  LocalOp ConvertElementType(const LocalOp& operand,
                             PrimitiveType new_element_type);

  LocalOp BitcastConvertType(const LocalOp& operand,
                             PrimitiveType new_element_type);

  LocalOp Call(const LocalComputation& local_computation,
               absl::Span<const LocalOp> operands);

  LocalOp CustomCall(const string& call_target_name,
                     absl::Span<const LocalOp> operands,
                     const Shape& shape_with_layout,
                     const std::vector<Shape>& operand_shapes_with_layout,
                     const string& opaque);

  LocalOp Transpose(const LocalOp& operand,
                    absl::Span<const int64> permutation);

  LocalOp Rev(const LocalOp& operand, absl::Span<const int64> dimensions);

  LocalOp Map(absl::Span<const LocalOp> operands,
              const LocalComputation& local_computation,
              absl::Span<const int64> dimensions);

  LocalOp Reduce(const LocalOp& operand, const LocalOp& init_value,
                 const LocalComputation& local_computation,
                 absl::Span<const int64> dimensions_to_reduce);

  LocalOp ReduceWindowWithGeneralPadding(
      const LocalOp& operand, const LocalOp& init_value,
      const LocalComputation& local_computation,
      absl::Span<const int64> window_dimensions,
      absl::Span<const int64> window_strides,
      absl::Span<const int64> base_dilations,
      absl::Span<const int64> window_dilations,
      absl::Span<const std::pair<int64, int64> > padding);

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

  LocalOp Sort(const LocalOp& operand, int64 dimension);

  LocalOp SortKeyVal(const LocalOp& keys, const LocalOp& values,
                     int64 dimension);

  LocalOp QR(const LocalOp& a, bool full_matrices);

  LocalOp Cholesky(const LocalOp& a);

  LocalOp TriangularSolve(const LocalOp& a, const LocalOp& b, bool left_side,
                          bool lower, bool transpose_a, bool conjugate_a,
                          bool unit_diagonal);

  LocalOp Gather(const LocalOp& input, const LocalOp& start_indices,
                 const GatherDimensionNumbers& dimension_numbers,
                 absl::Span<const int64> slice_sizes);

  LocalOp Scatter(const LocalOp& input, const LocalOp& scatter_indices,
                  const LocalOp& updates,
                  const LocalComputation& update_computation,
                  const ScatterDimensionNumbers& dimension_numbers);

  StatusOr<LocalComputation*> BuildConstantSubGraph(const LocalOp& operand);

#define _FORWARD(method_name, return_sig, args_sig) \
  return_sig method_name args_sig;

#define _FORWARD_UNOP(method_name) \
  _FORWARD(method_name, LocalOp, (const LocalOp& operand))

#define _FORWARD_BINOP(method_name)                 \
  _FORWARD(method_name, LocalOp,                    \
           (const LocalOp& lhs, const LocalOp& rhs, \
            absl::Span<const int64> broadcast_dimensions))

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
  _FORWARD_BINOP(Complex)
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
  _FORWARD_UNOP(Real)
  _FORWARD_UNOP(Imag)
  _FORWARD_UNOP(Conj)

#undef _FORWARD
#undef _FORWARD_UNOP
#undef _FORWARD_BINOP
#undef _FORWARD_TRIOP

 private:
  XlaBuilder builder_;
};

// Functions for freeing resources from the Python side.
void DeleteLocalShapedBuffer(LocalShapedBuffer* local_shaped_buffer);
void DeleteXrtAllocation(XrtAllocation* allocation);
void DeleteCompiledLocalComputation(CompiledLocalComputation* computation);
void DeleteCompiledXrtComputation(CompiledXrtComputation* computation);
void DeleteLocalComputation(LocalComputation* computation);

}  // namespace swig
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_COMPUTATION_BUILDER_H_
