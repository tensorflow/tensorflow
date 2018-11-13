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

#include "tensorflow/compiler/xla/python/local_computation_builder.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {
namespace swig {

// TODO(b/34473877) Ideally XLA would support AllReduce among arbitrary sets of
// device handles instead of needing to set the number of replicas at XLA
// service initialization time.
tensorflow::mutex g_local_client_mutex(tensorflow::LINKER_INITIALIZED);
int g_replica_count GUARDED_BY(g_local_client_mutex) = 1;
LocalClient* g_local_client GUARDED_BY(g_local_client_mutex) = nullptr;

Status InitializeReplicaCount(int replica_count) {
  if (replica_count < 1) {
    return InvalidArgument("Replica count must be >= 1; got %d.",
                           replica_count);
  }
  tensorflow::mutex_lock lock(g_local_client_mutex);
  if (g_local_client != nullptr) {
    return FailedPrecondition(
        "Attempted to set the replica count to %d, but a local XLA service was "
        "previously created with a replica count of %d.",
        replica_count, g_replica_count);
  }
  g_replica_count = replica_count;
  return Status::OK();
}

int GetReplicaCount() {
  tensorflow::mutex_lock lock(g_local_client_mutex);
  return g_replica_count;
}

LocalClient* GetOrCreateLocalClient() {
  tensorflow::mutex_lock lock(g_local_client_mutex);
  if (g_local_client != nullptr) {
    return g_local_client;
  }
  LocalClientOptions options;
  options.set_number_of_replicas(g_replica_count);
  g_local_client = ClientLibrary::GetOrCreateLocalClient(options).ValueOrDie();
  CHECK(g_local_client != nullptr);
  return g_local_client;
}

Status TransferToInfeedLocal(const Literal& literal) {
  VLOG(1) << "Infeeding literal without replica number; shape: "
          << literal.shape();
  LocalClient* client = GetOrCreateLocalClient();
  return client->TransferToInfeedLocal(literal, /*device_ordinal=*/0);
}

Status TransferToInfeedLocalReplica(const Literal& literal,
                                    int replica_number) {
  VLOG(1) << "Infeeding shape " << literal.shape()
          << " to replica number: " << replica_number;
  LocalClient* client = GetOrCreateLocalClient();
  TF_ASSIGN_OR_RETURN(int device_ordinal,
                      client->ReplicaNumberToDeviceOrdinal(replica_number));
  return client->TransferToInfeedLocal(literal, device_ordinal);
}

StatusOr<std::unique_ptr<Literal>> TransferFromOutfeedLocalReplica(
    const Shape& shape, int replica_number) {
  VLOG(1) << "Outfeeding literal from replica number: " << replica_number
          << " shape: " << shape;
  LocalClient* client = GetOrCreateLocalClient();
  TF_ASSIGN_OR_RETURN(int device_ordinal,
                      client->ReplicaNumberToDeviceOrdinal(replica_number));
  return client->TransferFromOutfeedLocal(shape, device_ordinal);
}

LocalShapedBuffer::LocalShapedBuffer(ScopedShapedBuffer shaped_buffer)
    : shaped_buffer_(std::move(shaped_buffer)) {}

const ScopedShapedBuffer* LocalShapedBuffer::shaped_buffer() const {
  return &shaped_buffer_;
}

ShapedBuffer LocalShapedBuffer::Release() { return shaped_buffer_.release(); }

LocalShapedBufferTuple::LocalShapedBufferTuple(
    std::vector<LocalShapedBuffer*> elements)
    : elements_(std::move(elements)) {
  for (auto* element : elements_) {
    DCHECK(element != nullptr);
  }
}

LocalShapedBufferTuple::~LocalShapedBufferTuple() {
  for (LocalShapedBuffer* element : elements_) {
    if (element != nullptr) {
      delete element;
    }
  }
}

StatusOr<LocalShapedBuffer*> LocalShapedBufferTuple::Release(int i) {
  LocalShapedBuffer* element = elements_[i];
  if (element == nullptr) {
    return InvalidArgument("Attempted to release already-released element %d.",
                           i);
  }
  elements_[i] = nullptr;
  return element;
}

int LocalShapedBufferTuple::size() const { return elements_.size(); }

static StatusOr<ScopedShapedBuffer> ToBuffer(LocalClient* client,
                                             int device_ordinal,
                                             const Literal& arg) {
  return client->LiteralToShapedBuffer(arg, device_ordinal,
                                       client->backend().memory_allocator());
}

/* static */
StatusOr<LocalShapedBuffer*> LocalShapedBuffer::FromLiteral(
    const Literal& argument, const absl::optional<Shape>& shape_with_layout) {
  LocalClient* client = GetOrCreateLocalClient();
  StatusOr<ScopedShapedBuffer> buf = [&] {
    if (shape_with_layout) {
      std::unique_ptr<Literal> relaid =
          argument.Relayout(shape_with_layout.value());
      return ToBuffer(client, /*device_ordinal=*/0, *relaid);
    }
    return ToBuffer(client, /*device_ordinal=*/0, argument);
  }();
  TF_RETURN_IF_ERROR(buf.status());
  return new LocalShapedBuffer(std::move(buf).ValueOrDie());
}

StatusOr<std::unique_ptr<Literal>> LocalShapedBuffer::ToLiteral() const {
  LocalClient* client = GetOrCreateLocalClient();
  return client->ShapedBufferToLiteral(*shaped_buffer());
}

CompiledLocalComputation::CompiledLocalComputation(
    std::unique_ptr<LocalExecutable> executable)
    : executable_(std::move(executable)) {}

StatusOr<std::unique_ptr<Literal>> CompiledLocalComputation::Execute(
    const std::vector<Literal>& arguments,
    const std::vector<absl::optional<Shape>>& shapes_with_layout) {
  LocalClient* client = GetOrCreateLocalClient();

  VLOG(1) << "Execution requested with " << GetReplicaCount() << " replicas.";

  // Each replica populates a StatusOr result, but only replica zero actually
  // retrieves its literal value.
  std::vector<StatusOr<std::unique_ptr<Literal>>> results(GetReplicaCount());
  {
    tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "xlarun",
                                        GetReplicaCount());

    for (int replica = 0; replica < GetReplicaCount(); ++replica) {
      pool.Schedule(
          [this, client, replica, &arguments, &shapes_with_layout, &results] {
            StatusOr<int> device_ordinal_status =
                client->ReplicaNumberToDeviceOrdinal(replica);
            if (!device_ordinal_status.ok()) {
              results[replica] = device_ordinal_status.status();
              return;
            }
            const int device_ordinal = device_ordinal_status.ValueOrDie();
            VLOG(3) << "Replica " << replica
                    << " mapped to device ordinal for execution: "
                    << device_ordinal;

            // Transfer arguments in
            std::vector<ScopedShapedBuffer> scoped_buffers;
            scoped_buffers.reserve(arguments.size());
            for (int i = 0; i < arguments.size(); ++i) {
              const Literal& argument = arguments[i];
              const absl::optional<Shape>& shape_with_layout =
                  shapes_with_layout[i];

              StatusOr<ScopedShapedBuffer> pushed;
              if (shape_with_layout) {
                std::unique_ptr<Literal> relaid =
                    argument.Relayout(shape_with_layout.value());
                pushed = ToBuffer(client, device_ordinal, *relaid);
              } else {
                pushed = ToBuffer(client, device_ordinal, argument);
              }
              if (!pushed.ok()) {
                results[replica] = pushed.status();
                return;
              }

              scoped_buffers.push_back(std::move(pushed).ValueOrDie());
            }

            // Execute
            std::vector<const ShapedBuffer*> argument_buffers;
            argument_buffers.reserve(scoped_buffers.size());
            for (auto& buffer : scoped_buffers) {
              argument_buffers.push_back(&buffer);
            }

            DeviceAssignment device_assignment =
                client->backend()
                    .computation_placer()
                    ->AssignDevices(GetReplicaCount(), /*computation_count=*/1)
                    .ConsumeValueOrDie();

            ExecutableRunOptions options;
            options.set_device_ordinal(device_ordinal);
            options.set_allocator(client->backend().memory_allocator());
            options.set_intra_op_thread_pool(
                client->backend().eigen_intra_op_thread_pool_device());
            options.set_device_assignment(&device_assignment);
            StatusOr<ScopedShapedBuffer> result_buffer_status =
                executable_->Run(argument_buffers, options);
            if (!result_buffer_status.ok()) {
              results[replica] = result_buffer_status.status();
              return;
            }

            // Transfer result out
            results[replica] = client->ShapedBufferToLiteral(
                std::move(result_buffer_status).ValueOrDie());
          });
    }
  }

  for (int replica = 0; replica < GetReplicaCount(); ++replica) {
    const auto& statusor = results[replica];
    if (!statusor.ok()) {
      return InternalError(
          "Failed running replica %d (other replicas may have failed as well): "
          "%s.",
          replica, statusor.status().ToString());
    }
  }

  return std::move(results[0]);
}

LocalShapedBuffer* CompiledLocalComputation::ExecuteWithShapedBuffers(
    absl::Span<LocalShapedBuffer* const> argument_handles) {
  LocalClient* client = GetOrCreateLocalClient();

  std::vector<const ShapedBuffer*> argument_buffers;
  argument_buffers.reserve(argument_handles.size());
  for (auto& handle : argument_handles) {
    argument_buffers.push_back(handle->shaped_buffer());
  }

  // Execute
  ExecutableRunOptions options;
  options.set_allocator(client->backend().memory_allocator());
  options.set_intra_op_thread_pool(
      client->backend().eigen_intra_op_thread_pool_device());
  ScopedShapedBuffer result_buffer =
      executable_->Run(argument_buffers, options).ConsumeValueOrDie();

  return new LocalShapedBuffer(std::move(result_buffer));
}

LocalComputation::LocalComputation(XlaComputation computation)
    : computation_(std::move(computation)) {}

StatusOr<CompiledLocalComputation*> LocalComputation::Compile(
    const std::vector<Shape>& argument_shapes,
    const ExecutableBuildOptions* build_options) {
  std::vector<const Shape*> argument_shape_pointers;
  argument_shape_pointers.reserve(argument_shapes.size());
  for (auto& argument_shape : argument_shapes) {
    argument_shape_pointers.push_back(&argument_shape);
  }

  LocalClient* client = GetOrCreateLocalClient();
  ExecutableBuildOptions options;
  if (build_options != nullptr) {
    options = *build_options;
  }
  TF_ASSIGN_OR_RETURN(
      auto local_executable,
      client->Compile(computation_, argument_shape_pointers, options));
  return new CompiledLocalComputation(std::move(local_executable));
}

const XlaComputation& LocalComputation::computation() const {
  return computation_;
}

string LocalComputation::GetSerializedProto() const {
  string result;
  if (!computation_.proto().SerializeToString(&result)) {
    LOG(ERROR) << "Failed to serialize the HloModuleProto.";
    return "";
  }
  return result;
}

StatusOr<Shape> LocalComputation::GetReturnValueShape() const {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation_.GetProgramShape());
  return std::move(*program_shape.mutable_result());
}

LocalOp::LocalOp(const XlaOp& op) : op_(op) {}

const XlaOp& LocalOp::op() const { return op_; }

LocalComputationBuilder::LocalComputationBuilder(const string& computation_name)
    : builder_(computation_name) {}

void LocalComputationBuilder::SetOpMetadata(const OpMetadata& metadata) {
  builder_.SetOpMetadata(metadata);
}

void LocalComputationBuilder::ClearOpMetadata() { builder_.ClearOpMetadata(); }

StatusOr<LocalComputation*> LocalComputationBuilder::Build() {
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder_.Build());
  return new LocalComputation(std::move(computation));
}

LocalOp LocalComputationBuilder::Parameter(int64 parameter_number,
                                           const Shape& shape,
                                           const string& name) {
  return xla::Parameter(&builder_, parameter_number, shape, name);
}

StatusOr<Shape> LocalComputationBuilder::GetShape(const LocalOp& operand) {
  return builder_.GetShape(operand.op());
}

StatusOr<Shape> LocalComputationBuilder::GetReturnValueShape() {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, builder_.GetProgramShape());
  return program_shape.result();
}

LocalOp LocalComputationBuilder::Infeed(const Shape& shape) {
  return xla::Infeed(&builder_, shape);
}

void LocalComputationBuilder::Outfeed(const LocalOp& operand,
                                      const Shape& shape,
                                      const string& outfeed_config) {
  xla::Outfeed(operand.op(), shape, outfeed_config);
}

LocalOp LocalComputationBuilder::ConstantLiteral(const Literal& literal) {
  return xla::ConstantLiteral(&builder_, literal);
}

LocalOp LocalComputationBuilder::Broadcast(
    const LocalOp& operand, absl::Span<const int64> broadcast_sizes) {
  return xla::Broadcast(operand.op(), broadcast_sizes);
}

LocalOp LocalComputationBuilder::Pad(const LocalOp& operand,
                                     const LocalOp& padding_value,
                                     const PaddingConfig& padding_config) {
  return xla::Pad(operand.op(), padding_value.op(), padding_config);
}

LocalOp LocalComputationBuilder::Reshape(const LocalOp& operand,
                                         absl::Span<const int64> dimensions,
                                         absl::Span<const int64> new_sizes) {
  return xla::Reshape(operand.op(), dimensions, new_sizes);
}

LocalOp LocalComputationBuilder::Collapse(const LocalOp& operand,
                                          absl::Span<const int64> dimensions) {
  return xla::Collapse(operand.op(), dimensions);
}

LocalOp LocalComputationBuilder::CrossReplicaSum(const LocalOp& operand) {
  return xla::CrossReplicaSum(operand.op());
}

LocalOp LocalComputationBuilder::Slice(const LocalOp& operand,
                                       absl::Span<const int64> start_indices,
                                       absl::Span<const int64> limit_indices,
                                       absl::Span<const int64> strides) {
  return xla::Slice(operand.op(), start_indices, limit_indices, strides);
}

LocalOp LocalComputationBuilder::SliceInDim(const LocalOp& operand,
                                            int64 start_index,
                                            int64 limit_index, int64 stride,
                                            int64 dimno) {
  return xla::SliceInDim(operand.op(), start_index, limit_index, stride, dimno);
}

LocalOp LocalComputationBuilder::DynamicSlice(
    const LocalOp& operand, const LocalOp& start_indices,
    absl::Span<const int64> slice_sizes) {
  return xla::DynamicSlice(operand.op(), start_indices.op(), slice_sizes);
}

LocalOp LocalComputationBuilder::DynamicUpdateSlice(
    const LocalOp& operand, const LocalOp& update,
    const LocalOp& start_indices) {
  return xla::DynamicUpdateSlice(operand.op(), update.op(), start_indices.op());
}

LocalOp LocalComputationBuilder::ConcatInDim(absl::Span<const LocalOp> operands,
                                             int64 dimension) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }
  return xla::ConcatInDim(&builder_, xla_ops, dimension);
}

LocalOp LocalComputationBuilder::SelectAndScatterWithGeneralPadding(
    const LocalOp& operand, const LocalComputation& select,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding, const LocalOp& source,
    const LocalOp& init_value, const LocalComputation& scatter) {
  return xla::SelectAndScatterWithGeneralPadding(
      operand.op(), select.computation(), window_dimensions, window_strides,
      padding, source.op(), init_value.op(), scatter.computation());
}

LocalOp LocalComputationBuilder::Tuple(absl::Span<const LocalOp> elements) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(elements.size());
  for (const auto& op : elements) {
    xla_ops.push_back(op.op());
  }

  return xla::Tuple(&builder_, xla_ops);
}

LocalOp LocalComputationBuilder::GetTupleElement(const LocalOp& tuple_data,
                                                 int64 index) {
  return xla::GetTupleElement(tuple_data.op(), index);
}

LocalOp LocalComputationBuilder::Dot(const LocalOp& lhs, const LocalOp& rhs) {
  return xla::Dot(lhs.op(), rhs.op());
}

LocalOp LocalComputationBuilder::DotGeneral(
    const LocalOp& lhs, const LocalOp& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  return xla::DotGeneral(lhs.op(), rhs.op(), dimension_numbers);
}

LocalOp LocalComputationBuilder::ConvGeneralDilated(
    const LocalOp& lhs, const LocalOp& rhs,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    absl::Span<const int64> lhs_dilation, absl::Span<const int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return xla::ConvGeneralDilated(lhs.op(), rhs.op(), window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers);
}

LocalOp LocalComputationBuilder::ConvertElementType(
    const LocalOp& operand, PrimitiveType new_element_type) {
  return xla::ConvertElementType(operand.op(), new_element_type);
}

LocalOp LocalComputationBuilder::BitcastConvertType(
    const LocalOp& operand, PrimitiveType new_element_type) {
  return xla::BitcastConvertType(operand.op(), new_element_type);
}

LocalOp LocalComputationBuilder::Call(const LocalComputation& local_computation,
                                      absl::Span<const LocalOp> operands) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }
  return xla::Call(&builder_, local_computation.computation(), xla_ops);
}

LocalOp LocalComputationBuilder::Transpose(
    const LocalOp& operand, absl::Span<const int64> permutation) {
  return xla::Transpose(operand.op(), permutation);
}

LocalOp LocalComputationBuilder::Rev(const LocalOp& operand,
                                     absl::Span<const int64> dimensions) {
  return xla::Rev(operand.op(), dimensions);
}

LocalOp LocalComputationBuilder::Map(absl::Span<const LocalOp> operands,
                                     const LocalComputation& local_computation,
                                     absl::Span<const int64> dimensions) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }

  return xla::Map(&builder_, xla_ops, local_computation.computation(),
                  dimensions);
}

LocalOp LocalComputationBuilder::Reduce(
    const LocalOp& operand, const LocalOp& init_value,
    const LocalComputation& local_computation,
    absl::Span<const int64> dimensions_to_reduce) {
  return xla::Reduce(operand.op(), init_value.op(),
                     local_computation.computation(), dimensions_to_reduce);
}

LocalOp LocalComputationBuilder::ReduceWindowWithGeneralPadding(
    const LocalOp& operand, const LocalOp& init_value,
    const LocalComputation& local_computation,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding) {
  return xla::ReduceWindowWithGeneralPadding(
      operand.op(), init_value.op(), local_computation.computation(),
      window_dimensions, window_strides, padding);
}

LocalOp LocalComputationBuilder::RngNormal(const LocalOp& mu,
                                           const LocalOp& sigma,
                                           const Shape& shape) {
  return xla::RngNormal(mu.op(), sigma.op(), shape);
}

LocalOp LocalComputationBuilder::RngUniform(const LocalOp& a, const LocalOp& b,
                                            const Shape& shape) {
  return xla::RngUniform(a.op(), b.op(), shape);
}

LocalOp LocalComputationBuilder::While(const LocalComputation& condition,
                                       const LocalComputation& body,
                                       const LocalOp& init) {
  return xla::While(condition.computation(), body.computation(), init.op());
}

LocalOp LocalComputationBuilder::Conditional(
    const LocalOp& predicate, const LocalOp& true_operand,
    const LocalComputation& true_computation, const LocalOp& false_operand,
    const LocalComputation& false_computation) {
  return xla::Conditional(predicate.op(), true_operand.op(),
                          true_computation.computation(), false_operand.op(),
                          false_computation.computation());
}

StatusOr<bool> LocalComputationBuilder::IsConstant(const LocalOp& operand) {
  return builder_.IsConstant(operand.op());
}

LocalOp LocalComputationBuilder::Sort(const LocalOp& operand, int64 dimension) {
  return xla::Sort(operand.op(), absl::nullopt, dimension);
}

LocalOp LocalComputationBuilder::SortKeyVal(const LocalOp& keys,
                                            const LocalOp& values,
                                            int64 dimension) {
  return xla::Sort(keys.op(), values.op(), dimension);
}

StatusOr<LocalComputation*> LocalComputationBuilder::BuildConstantSubGraph(
    const LocalOp& operand) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation,
                      builder_.BuildConstantSubGraph(operand.op()));
  return new LocalComputation(std::move(computation));
}

#define _FORWARD(method_name, return_sig, args_sig, args)    \
  return_sig LocalComputationBuilder::method_name args_sig { \
    return xla::method_name args;                            \
  }

#define _FORWARD_UNOP(method_name) \
  _FORWARD(method_name, LocalOp, (const LocalOp& operand), (operand.op()))

#define _FORWARD_BINOP(method_name)                        \
  _FORWARD(method_name, LocalOp,                           \
           (const LocalOp& lhs, const LocalOp& rhs,        \
            absl::Span<const int64> broadcast_dimensions), \
           (lhs.op(), rhs.op(), broadcast_dimensions))

#define _FORWARD_TRIOP(method_name)                                      \
  _FORWARD(method_name, LocalOp,                                         \
           (const LocalOp& lhs, const LocalOp& rhs, const LocalOp& ehs), \
           (lhs.op(), rhs.op(), ehs.op()))

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

void DeleteLocalShapedBuffer(LocalShapedBuffer* local_shaped_buffer) {
  delete local_shaped_buffer;
}

void DeleteCompiledLocalComputation(CompiledLocalComputation* computation) {
  delete computation;
}

void DeleteLocalComputation(LocalComputation* computation) {
  delete computation;
}

StatusOr<LocalShapedBufferTuple*> DestructureLocalShapedBufferTuple(
    LocalShapedBuffer* local_shaped_buffer) {
  if (!ShapeUtil::IsTuple(
          local_shaped_buffer->shaped_buffer()->on_device_shape())) {
    return InvalidArgument(
        "Attemped to destructure a LocalShapedBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(
            local_shaped_buffer->shaped_buffer()->on_device_shape()));
  }

  DeviceMemoryAllocator* allocator =
      local_shaped_buffer->shaped_buffer()->memory_allocator();
  ShapedBuffer tuple_buffer = local_shaped_buffer->Release();

  // Extract some metadata we use to construct scoped buffers.
  const se::Platform* platform = tuple_buffer.platform();
  int device_ordinal = tuple_buffer.device_ordinal();

  ShapeTree<se::DeviceMemoryBase>& shape_tree = tuple_buffer.buffers();
  const Shape& tuple_shape = tuple_buffer.on_device_shape();
  std::vector<LocalShapedBuffer*> results;
  for (int64 i = 0; i < ShapeUtil::TupleElementCount(tuple_shape); ++i) {
    // Create a shaped buffer for this destructured tuple element.
    const Shape& subshape = ShapeUtil::GetSubshape(tuple_shape, {i});
    VLOG(3) << "Starting tuple element " << i << " subshape: " << subshape;
    ShapedBuffer shaped_buffer(subshape, subshape, platform, device_ordinal);

    ShapeUtil::ForEachSubshape(
        subshape, [&](const Shape& s, const ShapeIndex& index) {
          ShapeIndex original(index);
          original.push_front(i);
          se::DeviceMemoryBase* device_memory =
              shape_tree.mutable_element(original);
          shaped_buffer.set_buffer(*device_memory, index);
          *device_memory = se::DeviceMemoryBase();
        });

    VLOG(3) << "Completed tuple element: " << i;
    results.push_back(new LocalShapedBuffer(
        ScopedShapedBuffer(std::move(shaped_buffer), allocator)));
  }
  // Deallocate the root buffer.
  se::DeviceMemoryBase root_buffer = tuple_buffer.root_buffer();
  TF_RETURN_IF_ERROR(allocator->Deallocate(device_ordinal, root_buffer));
  return new LocalShapedBufferTuple(std::move(results));
}

}  // namespace swig
}  // namespace xla
