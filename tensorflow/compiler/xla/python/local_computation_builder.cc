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
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/default/thread_annotations.h"

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

LocalShapedBuffer::LocalShapedBuffer(
    std::unique_ptr<ScopedShapedBuffer> shaped_buffer)
    : shaped_buffer_(std::move(shaped_buffer)) {}

const std::unique_ptr<ScopedShapedBuffer>& LocalShapedBuffer::shaped_buffer()
    const {
  return shaped_buffer_;
}

static StatusOr<std::unique_ptr<ScopedShapedBuffer>> ToBuffer(
    LocalClient* client, int device_ordinal, const Literal& arg) {
  return client->LiteralToShapedBuffer(arg, device_ordinal,
                                       client->backend().memory_allocator());
}

/* static */
LocalShapedBuffer* LocalShapedBuffer::FromLiteral(
    const Literal& argument,
    const tensorflow::gtl::optional<Shape>& shape_with_layout) {
  LocalClient* client = GetOrCreateLocalClient();
  std::unique_ptr<ScopedShapedBuffer> buf;
  if (shape_with_layout) {
    std::unique_ptr<Literal> relaid =
        argument.Relayout(shape_with_layout.value());
    buf = ToBuffer(client, /*device_ordinal=*/0, *relaid).ConsumeValueOrDie();
  } else {
    buf = ToBuffer(client, /*device_ordinal=*/0, argument).ConsumeValueOrDie();
  }
  return new LocalShapedBuffer(std::move(buf));
}

std::unique_ptr<Literal> LocalShapedBuffer::ToLiteral() const {
  LocalClient* client = GetOrCreateLocalClient();
  return client->ShapedBufferToLiteral(*shaped_buffer()).ConsumeValueOrDie();
}

CompiledLocalComputation::CompiledLocalComputation(
    std::unique_ptr<LocalExecutable> executable)
    : executable_(std::move(executable)) {}

StatusOr<std::unique_ptr<Literal>> CompiledLocalComputation::Execute(
    const std::vector<Literal>& arguments,
    const std::vector<tensorflow::gtl::optional<Shape>>& shapes_with_layout) {
  LocalClient* client = GetOrCreateLocalClient();

  VLOG(1) << "Execution requested with " << GetReplicaCount() << " replicas.";

  // Each replica populates a StatusOr result, but only replica zero actually
  // retrieves its literal value.
  std::vector<StatusOr<std::unique_ptr<Literal>>> results(GetReplicaCount());
  {
    tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "xlarun",
                                        GetReplicaCount());

    for (int replica = 0; replica < GetReplicaCount(); ++replica) {
      pool.Schedule([this, client, replica, &arguments, &shapes_with_layout,
                     &results] {
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
        std::vector<std::unique_ptr<ScopedShapedBuffer>> scoped_buffers;
        scoped_buffers.reserve(arguments.size());
        for (int i = 0; i < arguments.size(); ++i) {
          const Literal& argument = arguments[i];
          const tensorflow::gtl::optional<Shape>& shape_with_layout =
              shapes_with_layout[i];

          StatusOr<std::unique_ptr<ScopedShapedBuffer>> pushed;
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
          argument_buffers.push_back(buffer.get());
        }

        DeviceAssignment device_assignment =
            client->backend()
                .computation_placer()
                ->AssignDevices(GetReplicaCount(), /*computation_count=*/1)
                .ConsumeValueOrDie();

        ExecutableRunOptions options;
        options.set_device_ordinal(device_ordinal);
        options.set_allocator(client->backend().memory_allocator());
        options.set_inter_op_thread_pool(
            client->backend().inter_op_thread_pool());
        options.set_intra_op_thread_pool(
            client->backend().eigen_intra_op_thread_pool_device());
        options.set_device_assignment(&device_assignment);
        StatusOr<std::unique_ptr<ScopedShapedBuffer>> result_buffer_status =
            executable_->Run(argument_buffers, options);
        if (!result_buffer_status.ok()) {
          results[replica] = result_buffer_status.status();
          return;
        }

        // Transfer result out
        results[replica] =
            client->ShapedBufferToLiteral(*result_buffer_status.ValueOrDie());
      });
    }
  }

  for (int replica = 0; replica < GetReplicaCount(); ++replica) {
    const auto& statusor = results[replica];
    if (!statusor.ok()) {
      return InternalError(
          "Failed running replica %d (other replicas may have failed as well): "
          "%s.",
          replica, statusor.status().ToString().c_str());
    }
  }

  return std::move(results[0]);
}

LocalShapedBuffer* CompiledLocalComputation::ExecuteWithShapedBuffers(
    tensorflow::gtl::ArraySlice<LocalShapedBuffer*> argument_handles) {
  LocalClient* client = GetOrCreateLocalClient();

  std::vector<const ShapedBuffer*> argument_buffers;
  argument_buffers.reserve(argument_handles.size());
  for (auto& handle : argument_handles) {
    argument_buffers.push_back(handle->shaped_buffer().get());
  }

  // Execute
  ExecutableRunOptions options;
  options.set_allocator(client->backend().memory_allocator());
  options.set_inter_op_thread_pool(client->backend().inter_op_thread_pool());
  options.set_intra_op_thread_pool(
      client->backend().eigen_intra_op_thread_pool_device());
  std::unique_ptr<ScopedShapedBuffer> result_buffer =
      executable_->Run(argument_buffers, options).ConsumeValueOrDie();

  return new LocalShapedBuffer(std::move(result_buffer));
}

LocalComputation::LocalComputation(Computation computation)
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

const Computation& LocalComputation::computation() const {
  return computation_;
}

StatusOr<Shape> LocalComputation::GetReturnValueShape() const {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation_.GetProgramShape());
  return std::move(*program_shape.mutable_result());
}

LocalComputationBuilder::LocalComputationBuilder(const string& computation_name)
    : builder_(GetOrCreateLocalClient(), computation_name) {}

void LocalComputationBuilder::SetOpMetadata(const OpMetadata& metadata) {
  builder_.SetOpMetadata(metadata);
}

void LocalComputationBuilder::ClearOpMetadata() { builder_.ClearOpMetadata(); }

StatusOr<LocalComputation*> LocalComputationBuilder::Build() {
  TF_ASSIGN_OR_RETURN(Computation computation, builder_.Build());
  return new LocalComputation(std::move(computation));
}

ComputationDataHandle LocalComputationBuilder::Parameter(int64 parameter_number,
                                                         const Shape& shape,
                                                         const string& name) {
  return builder_.Parameter(parameter_number, shape, name);
}

std::unique_ptr<Shape> LocalComputationBuilder::GetShape(
    const ComputationDataHandle& operand) {
  return builder_.GetShape(operand).ConsumeValueOrDie();
}

StatusOr<Shape> LocalComputationBuilder::GetReturnValueShape() {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, builder_.GetProgramShape());
  return program_shape.result();
}

ComputationDataHandle LocalComputationBuilder::Infeed(const Shape& shape) {
  return builder_.Infeed(shape);
}

void LocalComputationBuilder::Outfeed(const ComputationDataHandle& operand,
                                      const Shape& shape,
                                      const string& outfeed_config) {
  builder_.Outfeed(operand, shape, outfeed_config);
}

ComputationDataHandle LocalComputationBuilder::ConstantLiteral(
    const Literal& literal) {
  return builder_.ConstantLiteral(literal);
}

ComputationDataHandle LocalComputationBuilder::Broadcast(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_sizes) {
  return builder_.Broadcast(operand, broadcast_sizes);
}

ComputationDataHandle LocalComputationBuilder::Pad(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& padding_value,
    const PaddingConfig& padding_config) {
  return builder_.Pad(operand, padding_value, padding_config);
}

ComputationDataHandle LocalComputationBuilder::Reshape(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::gtl::ArraySlice<int64> new_sizes) {
  return builder_.Reshape(operand, dimensions, new_sizes);
}

ComputationDataHandle LocalComputationBuilder::Collapse(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  return builder_.Collapse(operand, dimensions);
}

ComputationDataHandle LocalComputationBuilder::CrossReplicaSum(
    const ComputationDataHandle& operand) {
  return builder_.CrossReplicaSum(operand);
}

ComputationDataHandle LocalComputationBuilder::Slice(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides) {
  return builder_.Slice(operand, start_indices, limit_indices, strides);
}

ComputationDataHandle LocalComputationBuilder::SliceInDim(
    const ComputationDataHandle& operand, int64 start_index, int64 limit_index,
    int64 stride, int64 dimno) {
  return builder_.SliceInDim(operand, start_index, limit_index, stride, dimno);
}

ComputationDataHandle LocalComputationBuilder::DynamicSlice(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& start_indices,
    tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  return builder_.DynamicSlice(operand, start_indices, slice_sizes);
}

ComputationDataHandle LocalComputationBuilder::DynamicUpdateSlice(
    const ComputationDataHandle& operand, const ComputationDataHandle& update,
    const ComputationDataHandle& start_indices) {
  return builder_.DynamicUpdateSlice(operand, update, start_indices);
}

ComputationDataHandle LocalComputationBuilder::ConcatInDim(
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
    int64 dimension) {
  return builder_.ConcatInDim(operands, dimension);
}

ComputationDataHandle
LocalComputationBuilder::SelectAndScatterWithGeneralPadding(
    const ComputationDataHandle& operand, const LocalComputation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const ComputationDataHandle& source,
    const ComputationDataHandle& init_value, const LocalComputation& scatter) {
  return builder_.SelectAndScatterWithGeneralPadding(
      operand, select.computation(), window_dimensions, window_strides, padding,
      source, init_value, scatter.computation());
}

ComputationDataHandle LocalComputationBuilder::Tuple(
    tensorflow::gtl::ArraySlice<ComputationDataHandle> elements) {
  return builder_.Tuple(elements);
}

ComputationDataHandle LocalComputationBuilder::GetTupleElement(
    const ComputationDataHandle& tuple_data, int64 index) {
  return builder_.GetTupleElement(tuple_data, index);
}

ComputationDataHandle LocalComputationBuilder::Dot(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs) {
  return builder_.Dot(lhs, rhs);
}

ComputationDataHandle LocalComputationBuilder::DotGeneral(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  return builder_.DotGeneral(lhs, rhs, dimension_numbers);
}

ComputationDataHandle LocalComputationBuilder::ConvGeneralDilated(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return builder_.ConvGeneralDilated(lhs, rhs, window_strides, padding,
                                     lhs_dilation, rhs_dilation,
                                     dimension_numbers);
}

ComputationDataHandle LocalComputationBuilder::ConvertElementType(
    const ComputationDataHandle& operand, PrimitiveType new_element_type) {
  return builder_.ConvertElementType(operand, new_element_type);
}

ComputationDataHandle LocalComputationBuilder::Call(
    const LocalComputation& local_computation,
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands) {
  return builder_.Call(local_computation.computation(), operands);
}

ComputationDataHandle LocalComputationBuilder::Transpose(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> permutation) {
  return builder_.Transpose(operand, permutation);
}

ComputationDataHandle LocalComputationBuilder::Rev(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  return builder_.Rev(operand, dimensions);
}

ComputationDataHandle LocalComputationBuilder::Map(
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
    const LocalComputation& local_computation,
    tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::gtl::ArraySlice<ComputationDataHandle> static_operands) {
  return builder_.Map(operands, local_computation.computation(), dimensions,
                      static_operands);
}

ComputationDataHandle LocalComputationBuilder::Reduce(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& init_value,
    const LocalComputation& local_computation,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce) {
  return builder_.Reduce(operand, init_value, local_computation.computation(),
                         dimensions_to_reduce);
}

ComputationDataHandle LocalComputationBuilder::ReduceWindowWithGeneralPadding(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& init_value,
    const LocalComputation& local_computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
  return builder_.ReduceWindowWithGeneralPadding(
      operand, init_value, local_computation.computation(), window_dimensions,
      window_strides, padding);
}

ComputationDataHandle LocalComputationBuilder::RngNormal(
    const ComputationDataHandle& mu, const ComputationDataHandle& sigma,
    const Shape& shape) {
  return builder_.RngNormal(mu, sigma, shape);
}

ComputationDataHandle LocalComputationBuilder::RngUniform(
    const ComputationDataHandle& a, const ComputationDataHandle& b,
    const Shape& shape) {
  return builder_.RngUniform(a, b, shape);
}

ComputationDataHandle LocalComputationBuilder::While(
    const LocalComputation& condition, const LocalComputation& body,
    const ComputationDataHandle& init) {
  return builder_.While(condition.computation(), body.computation(), init);
}

ComputationDataHandle LocalComputationBuilder::Conditional(
    const ComputationDataHandle& predicate,
    const ComputationDataHandle& true_operand,
    const LocalComputation& true_computation,
    const ComputationDataHandle& false_operand,
    const LocalComputation& false_computation) {
  return builder_.Conditional(predicate, true_operand,
                              true_computation.computation(), false_operand,
                              false_computation.computation());
}

StatusOr<bool> LocalComputationBuilder::IsConstant(
    const ComputationDataHandle& operand, int64 num_parameters) {
  return builder_.IsConstant(operand, num_parameters);
}

StatusOr<std::unique_ptr<Literal>> LocalComputationBuilder::ComputeConstant(
    const ComputationDataHandle& operand, const Layout* output_layout,
    tensorflow::gtl::ArraySlice<Literal> parameters) {
  return builder_.ComputeConstant(operand, output_layout, parameters);
}

#define _FORWARD(method_name, return_sig, args_sig, args)    \
  return_sig LocalComputationBuilder::method_name args_sig { \
    return builder_.method_name args;                        \
  }

#define _FORWARD_UNOP(method_name)             \
  _FORWARD(method_name, ComputationDataHandle, \
           (const ComputationDataHandle& operand), (operand))

#define _FORWARD_BINOP(method_name)                                        \
  _FORWARD(                                                                \
      method_name, ComputationDataHandle,                                  \
      (const ComputationDataHandle& lhs, const ComputationDataHandle& rhs, \
       tensorflow::gtl::ArraySlice<int64> broadcast_dimensions),           \
      (lhs, rhs, broadcast_dimensions))

#define _FORWARD_TRIOP(method_name)                                        \
  _FORWARD(                                                                \
      method_name, ComputationDataHandle,                                  \
      (const ComputationDataHandle& lhs, const ComputationDataHandle& rhs, \
       const ComputationDataHandle& ehs),                                  \
      (lhs, rhs, ehs))

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

void DeleteLocalShapedBuffer(LocalShapedBuffer* local_shaped_buffer) {
  delete local_shaped_buffer;
}

void DeleteCompiledLocalComputation(CompiledLocalComputation* computation) {
  delete computation;
}

void DeleteLocalComputation(LocalComputation* computation) {
  delete computation;
}

}  // namespace swig

}  // namespace xla
