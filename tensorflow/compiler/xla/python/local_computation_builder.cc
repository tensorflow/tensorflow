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

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/client/lib/cholesky.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/triangular_solve.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_compile_ops.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_execute_op.h"
#include "tensorflow/compiler/xrt/cc/ops/xrt_state_ops.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace swig {

// TODO(b/118641336): Factor out XRT parts into a small c++ library of their
// own.

// TODO(b/34473877) Ideally XLA would support AllReduce among arbitrary sets of
// device handles instead of needing to set the number of replicas at XLA
// service initialization time.
tensorflow::mutex g_local_client_mutex(tensorflow::LINKER_INITIALIZED);
int g_replica_count GUARDED_BY(g_local_client_mutex) = 1;
LocalClient* g_local_client GUARDED_BY(g_local_client_mutex) = nullptr;

string* GetPlatformNameString() {
  static string* platform_name_string PT_GUARDED_BY(g_local_client_mutex) =
      new string("Host");
  return platform_name_string;
}

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

Status InitializePlatformName(const string& platform_name) {
  string* g_platform_name = GetPlatformNameString();
  tensorflow::mutex_lock lock(g_local_client_mutex);
  if (g_local_client != nullptr) {
    return FailedPrecondition(
        "Attempted to set the platform name to %s, but a local XLA service was "
        "previously created with a platform name of %s.",
        platform_name, *g_platform_name);
  }
  TF_RETURN_IF_ERROR(PlatformUtil::GetPlatform(platform_name).status());
  *g_platform_name = platform_name;
  return Status::OK();
}

int GetReplicaCount() {
  tensorflow::mutex_lock lock(g_local_client_mutex);
  return g_replica_count;
}

StatusOr<LocalClient*> GetOrCreateLocalClient() {
  string* platform_name = GetPlatformNameString();
  tensorflow::mutex_lock lock(g_local_client_mutex);
  if (g_local_client != nullptr) {
    return g_local_client;
  }
  LocalClientOptions options;
  options.set_platform(PlatformUtil::GetPlatform(*platform_name).ValueOrDie());
  options.set_number_of_replicas(g_replica_count);
  TF_ASSIGN_OR_RETURN(g_local_client,
                      ClientLibrary::GetOrCreateLocalClient(options));
  CHECK(g_local_client != nullptr);
  return g_local_client;
}

Status RegisterCpuCustomCallTarget(const string& fn_name, PyObject* capsule) {
  const char* name = "xla._CPU_CUSTOM_CALL_TARGET";
  if (!PyCapsule_IsValid(capsule, name)) {
    return InvalidArgument(
        "Argument to RegisterCpuCustomCallTargetRegistry was not a "
        "xla._CPU_CUSTOM_CALL_TARGET capsule.");
  }
  void* fn_ptr = PyCapsule_GetPointer(capsule, name);
  CHECK(fn_ptr != nullptr);
  cpu::CustomCallTargetRegistry::Global()->Register(
      std::string(fn_name.begin(), fn_name.end()), fn_ptr);
  return Status::OK();
}

Status TransferToInfeedLocal(const Literal& literal) {
  VLOG(1) << "Infeeding literal without replica number; shape: "
          << literal.shape();
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  return client->TransferToInfeedLocal(literal, /*device_ordinal=*/0);
}

Status TransferToInfeedLocalReplica(const Literal& literal,
                                    int replica_number) {
  VLOG(1) << "Infeeding shape " << literal.shape()
          << " to replica number: " << replica_number;
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  TF_ASSIGN_OR_RETURN(int device_ordinal,
                      client->ReplicaNumberToDeviceOrdinal(replica_number));
  return client->TransferToInfeedLocal(literal, device_ordinal);
}

StatusOr<Literal> TransferFromOutfeedLocalReplica(const Shape& shape,
                                                  int replica_number) {
  VLOG(1) << "Outfeeding literal from replica number: " << replica_number
          << " shape: " << shape;
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  TF_ASSIGN_OR_RETURN(int device_ordinal,
                      client->ReplicaNumberToDeviceOrdinal(replica_number));
  return client->TransferFromOutfeedLocal(shape, device_ordinal);
}

static StatusOr<ScopedShapedBuffer> ToBuffer(LocalClient* client,
                                             int device_ordinal,
                                             const Literal& arg) {
  return client->LiteralToShapedBuffer(arg, device_ordinal,
                                       client->backend().memory_allocator());
}

/* static */
StatusOr<LocalShapedBuffer*> LocalShapedBuffer::FromLiteral(
    const Literal& argument, const absl::optional<Shape>& shape_with_layout,
    int replica_number) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  TF_ASSIGN_OR_RETURN(int device_ordinal,
                      client->ReplicaNumberToDeviceOrdinal(replica_number));
  VLOG(1) << "Creating shaped buffer from literal on replica/ordinal: "
          << replica_number << "/" << device_ordinal;
  StatusOr<ScopedShapedBuffer> buf = [&] {
    if (shape_with_layout) {
      Literal relaid = argument.Relayout(shape_with_layout.value());
      return ToBuffer(client, device_ordinal, relaid);
    }
    return ToBuffer(client, device_ordinal, argument);
  }();
  TF_RETURN_IF_ERROR(buf.status());
  return new LocalShapedBuffer(std::move(buf).ValueOrDie());
}

LocalShapedBuffer::LocalShapedBuffer(ScopedShapedBuffer shaped_buffer)
    : shaped_buffer_(std::move(shaped_buffer)) {}

const ScopedShapedBuffer* LocalShapedBuffer::shaped_buffer() const {
  return &shaped_buffer_;
}

ShapedBuffer LocalShapedBuffer::Release() { return shaped_buffer_.release(); }

const Shape& LocalShapedBuffer::shape() const {
  return shaped_buffer()->on_device_shape();
}

StatusOr<Literal> LocalShapedBuffer::ToLiteral() const {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  return client->ShapedBufferToLiteral(*shaped_buffer());
}

LocalShapedBufferTuple::LocalShapedBufferTuple(
    std::vector<LocalShapedBuffer*> elements)
    : elements_(std::move(elements)) {
  for (auto* element : elements_) {
    CHECK(element != nullptr);
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

int64 LocalShapedBufferTuple::size() const { return elements_.size(); }

XrtAllocation::XrtAllocation(int64 handle, Shape shape,
                             const string& session_target)
    : handle_(handle), shape_(shape), session_target_(session_target) {}

XrtAllocation::~XrtAllocation() {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto allocation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto release =
      tensorflow::ops::XRTReleaseAllocationHandle(root, allocation_handle);
  if (!root.status().ok()) {
    LOG(ERROR) << root.status();
    return;
  }

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({allocation_handle, handle()});
  std::vector<tensorflow::Tensor> outputs;
  auto status = session.Run(inputs, {}, {release}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return;
  }
}

/* static */
StatusOr<XrtAllocation*> XrtAllocation::FromLiteral(
    const Literal& argument, const string& session_target) {
  xrt::XLAAllocation alloc;
  *alloc.mutable_value() = argument.ToProto();

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto literal_string =
      tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
  auto literal_handle = tensorflow::ops::XRTAllocate(root, literal_string);
  TF_RETURN_IF_ERROR(root.status());

  tensorflow::ClientSession session(root, session_target);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({literal_string, alloc.SerializeAsString()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {literal_handle}, &outputs));

  int64 handle = outputs[0].scalar<int64>()();
  return new XrtAllocation(handle, argument.shape(), session_target);
}

const int64 XrtAllocation::handle() const { return handle_; }

const Shape& XrtAllocation::shape() const { return shape_; }

StatusOr<Literal> XrtAllocation::ToLiteral() const {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto allocation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto read_literal = tensorflow::ops::XRTReadLiteral(root, allocation_handle);
  TF_RETURN_IF_ERROR(root.status());

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({allocation_handle, handle()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {read_literal}, &outputs));

  xla::LiteralProto response;
  TF_RET_CHECK(response.ParseFromString(outputs[0].scalar<string>()()));
  return Literal::CreateFromProto(response);
}

XrtAllocationTuple::XrtAllocationTuple(std::vector<XrtAllocation*> elements)
    : elements_(std::move(elements)) {
  for (auto* element : elements_) {
    CHECK(element != nullptr);
  }
}

XrtAllocationTuple::~XrtAllocationTuple() {
  for (XrtAllocation* element : elements_) {
    if (element != nullptr) {
      delete element;
    }
  }
}

StatusOr<XrtAllocation*> XrtAllocationTuple::Release(int i) {
  XrtAllocation* element = elements_[i];
  if (element == nullptr) {
    return InvalidArgument("Attempted to release already-released element %d.",
                           i);
  }
  elements_[i] = nullptr;
  return element;
}

int64 XrtAllocationTuple::size() const { return elements_.size(); }

CompiledLocalComputation::CompiledLocalComputation(
    std::unique_ptr<LocalExecutable> executable)
    : executable_(std::move(executable)) {}

StatusOr<LocalShapedBuffer*> CompiledLocalComputation::Execute(
    absl::Span<LocalShapedBuffer* const> argument_handles) {
  if (num_replicas() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d replicas using Execute()",
        num_replicas());
  }
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      client->backend().computation_placer()->AssignDevices(
                          1, /*computation_count=*/1));
  StatusOr<ScopedShapedBuffer> result_buffer_status;
  const int device_ordinal = device_assignment(0, 0);
  VLOG(3) << "Replica 0 mapped to device ordinal for execution: "
          << device_ordinal;

  std::vector<const ShapedBuffer*> argument_buffers;
  argument_buffers.reserve(argument_handles.size());
  for (auto& handle : argument_handles) {
    argument_buffers.push_back(handle->shaped_buffer());
  }

  ExecutableRunOptions options;
  options.set_device_ordinal(device_ordinal);
  options.set_allocator(client->backend().memory_allocator());
  options.set_intra_op_thread_pool(
      client->backend().eigen_intra_op_thread_pool_device());
  options.set_device_assignment(&device_assignment);

  result_buffer_status = executable_->Run(argument_buffers, options);

  if (!result_buffer_status.ok()) {
    return InternalError(
        "Failed running replica 0 (other replicas may have failed as well): "
        "%s.",
        result_buffer_status.status().ToString());
  }
  return new LocalShapedBuffer(std::move(result_buffer_status).ValueOrDie());
}

StatusOr<LocalShapedBufferTuple*> CompiledLocalComputation::ExecutePerReplica(
    absl::Span<const std::vector<LocalShapedBuffer*>> argument_handles) {
  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  const int num_devices = client->device_count();

  if (argument_handles.size() != num_replicas()) {
    return InvalidArgument(
        "Attempted to execute with %d replicas when replica count is %d",
        argument_handles.size(), num_devices);
  }
  if (argument_handles.size() > num_devices) {
    return InvalidArgument(
        "Attempted to execute with %d replicas when device count is %d",
        argument_handles.size(), num_devices);
  }

  VLOG(1) << "Executing with " << num_replicas() << " replicas.";

  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      client->backend().computation_placer()->AssignDevices(
                          num_replicas(), /*computation_count=*/1));

  std::vector<StatusOr<ScopedShapedBuffer>> results(num_replicas());
  auto execute = [this, client, &device_assignment, &argument_handles,
                  &results](int replica) {
    const int device_ordinal = device_assignment(replica, 0);
    VLOG(3) << "Replica " << replica
            << " mapped to device ordinal for execution: " << device_ordinal;

    std::vector<const ShapedBuffer*> argument_buffers;
    argument_buffers.reserve(argument_handles[replica].size());
    for (auto& handle : argument_handles[replica]) {
      argument_buffers.push_back(handle->shaped_buffer());
    }

    ExecutableRunOptions options;
    options.set_device_ordinal(device_ordinal);
    options.set_allocator(client->backend().memory_allocator());
    options.set_intra_op_thread_pool(
        client->backend().eigen_intra_op_thread_pool_device());
    options.set_device_assignment(&device_assignment);
    StatusOr<ScopedShapedBuffer> result_buffer_status =
        executable_->Run(argument_buffers, options);

    results[replica] = std::move(result_buffer_status);
  };

  if (num_replicas() == 1) {
    // Fast-path if there is only one replica — run the computation on the
    // current thread.
    execute(0);
  } else {
    // TODO(phawkins): don't recreate the threadpool for each execution.
    tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "xlarun",
                                        num_replicas() - 1);

    for (int replica = 0; replica < num_replicas() - 1; ++replica) {
      pool.Schedule([&execute, replica] { execute(replica); });
    }
    execute(num_replicas() - 1);
  }

  std::vector<LocalShapedBuffer*> wrapped_results(num_replicas());
  for (int replica = 0; replica < num_replicas(); ++replica) {
    auto& statusor = results[replica];
    if (!statusor.ok()) {
      return InternalError(
          "Failed running replica %d (other replicas may have failed as well): "
          "%s.",
          replica, statusor.status().ToString());
    }
    wrapped_results[replica] =
        new LocalShapedBuffer(std::move(statusor).ValueOrDie());
  }

  return new LocalShapedBufferTuple(std::move(wrapped_results));
}

static StatusOr<Shape> GetReturnValueShape(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  return std::move(*program_shape.mutable_result());
}

CompiledXrtComputation::CompiledXrtComputation(
    const ProgramShape& program_shape, int64 handle,
    const string& session_target)
    : program_shape_(program_shape),
      handle_(handle),
      session_target_(session_target) {}

CompiledXrtComputation::~CompiledXrtComputation() {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto computation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto release =
      tensorflow::ops::XRTReleaseCompilationHandle(root, computation_handle);
  if (!root.status().ok()) {
    LOG(ERROR) << root.status();
    return;
  }

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({computation_handle, handle()});
  std::vector<tensorflow::Tensor> outputs;
  auto status = session.Run(inputs, {}, {release}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return;
  }
}

StatusOr<XrtAllocation*> CompiledXrtComputation::Execute(
    absl::Span<XrtAllocation* const> argument_handles) {
  const int num_expected_arguments = program_shape().parameters().size();

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  std::vector<tensorflow::Output> arguments;
  arguments.reserve(num_expected_arguments);
  for (int i = 0; i < num_expected_arguments; ++i) {
    arguments.push_back(
        tensorflow::ops::Placeholder(root, tensorflow::DT_INT64));
  }
  auto computation_handle =
      tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto execution_config =
      tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
  auto execute = tensorflow::ops::XRTExecute(root, computation_handle,
                                             execution_config, arguments);
  TF_RETURN_IF_ERROR(root.status());

  TF_RET_CHECK(argument_handles.size() == arguments.size());

  xrt::XRTExecutionConfig e;
  e.set_release_input_handles(false);
  e.set_release_compilation_handle(false);

  tensorflow::ClientSession session(root, session_target_);
  tensorflow::ClientSession::FeedType inputs;
  for (int i = 0; i < arguments.size(); ++i) {
    inputs.insert({arguments[i], argument_handles[i]->handle()});
  }
  inputs.insert({computation_handle, handle()});
  inputs.insert({execution_config, e.SerializeAsString()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {execute}, &outputs));

  int64 output = outputs[0].scalar<int64>()();
  return new XrtAllocation(output, program_shape().result(), session_target_);
}

const ProgramShape& CompiledXrtComputation::program_shape() const {
  return program_shape_;
}

int64 CompiledXrtComputation::handle() const { return handle_; }

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

  TF_ASSIGN_OR_RETURN(LocalClient * client, GetOrCreateLocalClient());
  ExecutableBuildOptions options;
  if (build_options != nullptr) {
    options = *build_options;
  }
  TF_ASSIGN_OR_RETURN(
      auto local_executable,
      client->Compile(computation_, argument_shape_pointers, options));
  return new CompiledLocalComputation(std::move(local_executable));
}

StatusOr<CompiledXrtComputation*> LocalComputation::CompileForXrt(
    const std::vector<Shape>& argument_shapes, const string& session_target) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto program = tensorflow::ops::Placeholder(root, tensorflow::DT_STRING);
  auto compile = tensorflow::ops::XRTCompile(root, program);
  TF_RETURN_IF_ERROR(root.status());

  xrt::XLAComputation c;
  auto config = c.mutable_config();
  ProgramShape shapes;
  for (auto& shape : argument_shapes) {
    *shapes.add_parameters() = shape;
  }
  TF_ASSIGN_OR_RETURN(*shapes.mutable_result(), GetReturnValueShape());
  LayoutUtil::SetToDefaultLayout(&shapes);
  *config->mutable_program_shape() = shapes.ToProto();
  auto snapshot = computation().Snapshot().ValueOrDie();
  *c.mutable_hlo_snapshot() = *snapshot;

  tensorflow::ClientSession session(root, session_target);
  tensorflow::ClientSession::FeedType inputs;
  inputs.insert({program, c.SerializeAsString()});
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session.Run(inputs, {compile.handle}, &outputs));

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation().GetProgramShape());
  int64 handle = outputs[0].scalar<int64>()();
  return new CompiledXrtComputation(program_shape, handle, session_target);
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
  return swig::GetReturnValueShape(computation_);
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

StatusOr<LocalComputation*> LocalComputationBuilder::BuildWithRoot(
    const LocalOp& root) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder_.Build(root.op()));
  return new LocalComputation(std::move(computation));
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

LocalOp LocalComputationBuilder::Iota(PrimitiveType element_type, int64 size) {
  return xla::Iota(&builder_, element_type, size);
}

LocalOp LocalComputationBuilder::BroadcastedIota(const Shape& shape,
                                                 int64 dimension) {
  return xla::Iota(&builder_, shape, dimension);
}

LocalOp LocalComputationBuilder::Broadcast(
    const LocalOp& operand, absl::Span<const int64> broadcast_sizes) {
  return xla::Broadcast(operand.op(), broadcast_sizes);
}

LocalOp LocalComputationBuilder::BroadcastInDim(
    const LocalOp& operand, absl::Span<const int64> out_dim_sizes,
    absl::Span<const int64> broadcast_dimensions) {
  return xla::BroadcastInDim(operand.op(), out_dim_sizes, broadcast_dimensions);
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

LocalOp LocalComputationBuilder::CrossReplicaSum(
    const LocalOp& operand, absl::Span<const ReplicaGroup> replica_groups) {
  return xla::CrossReplicaSum(operand.op(), replica_groups);
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
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64 feature_group_count) {
  return xla::ConvGeneralDilated(lhs.op(), rhs.op(), window_strides, padding,
                                 lhs_dilation, rhs_dilation, dimension_numbers,
                                 feature_group_count);
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

LocalOp LocalComputationBuilder::CustomCall(
    const string& call_target_name, absl::Span<const LocalOp> operands,
    const Shape& shape_with_layout,
    const std::vector<Shape>& operand_shapes_with_layout,
    const string& opaque) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }
  return xla::CustomCallWithLayout(&builder_, call_target_name, xla_ops,
                                   shape_with_layout,
                                   operand_shapes_with_layout, opaque);
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
    absl::Span<const int64> base_dilations,
    absl::Span<const int64> window_dilations,
    absl::Span<const std::pair<int64, int64>> padding) {
  return xla::ReduceWindowWithGeneralPadding(
      operand.op(), init_value.op(), local_computation.computation(),
      window_dimensions, window_strides, base_dilations, window_dilations,
      padding);
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
  return xla::Sort(operand.op(), {}, dimension);
}

LocalOp LocalComputationBuilder::SortKeyVal(const LocalOp& keys,
                                            const LocalOp& values,
                                            int64 dimension) {
  return xla::Sort(keys.op(), {values.op()}, dimension);
}

LocalOp LocalComputationBuilder::Cholesky(const LocalOp& a) {
  return xla::Cholesky(a.op());
}

LocalOp LocalComputationBuilder::QR(const LocalOp& a, bool full_matrices) {
  XlaBuilder* builder = a.op().builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto qr, xla::QRDecomposition(a.op(), full_matrices));
    return xla::Tuple(builder, {qr.q, qr.r});
  });
}

LocalOp LocalComputationBuilder::TriangularSolve(const LocalOp& a,
                                                 const LocalOp& b,
                                                 bool left_side, bool lower,
                                                 bool transpose_a,
                                                 bool conjugate_a) {
  return xla::TriangularSolve(a.op(), b.op(), left_side, lower, transpose_a,
                              conjugate_a);
}

LocalOp LocalComputationBuilder::Gather(
    const LocalOp& input, const LocalOp& start_indices,
    const GatherDimensionNumbers& dimension_numbers,
    absl::Span<const int64> slice_sizes) {
  return xla::Gather(input.op(), start_indices.op(), dimension_numbers,
                     slice_sizes);
}

LocalOp LocalComputationBuilder::Scatter(
    const LocalOp& input, const LocalOp& scatter_indices,
    const LocalOp& updates, const LocalComputation& update_computation,
    const ScatterDimensionNumbers& dimension_numbers) {
  return xla::Scatter(input.op(), scatter_indices.op(), updates.op(),
                      update_computation.computation(), dimension_numbers);
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

void DeleteXrtAllocation(XrtAllocation* allocation) { delete allocation; }

void DeleteCompiledLocalComputation(CompiledLocalComputation* computation) {
  delete computation;
}

void DeleteCompiledXrtComputation(CompiledXrtComputation* computation) {
  delete computation;
}

void DeleteLocalComputation(LocalComputation* computation) {
  delete computation;
}

StatusOr<LocalShapedBufferTuple*> DestructureLocalShapedBufferTuple(
    LocalShapedBuffer* local_shaped_buffer) {
  const Shape tuple_shape = local_shaped_buffer->shape();

  if (!tuple_shape.IsTuple()) {
    return InvalidArgument(
        "Attemped to destructure a LocalShapedBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(tuple_shape));
  }

  DeviceMemoryAllocator* allocator =
      local_shaped_buffer->shaped_buffer()->memory_allocator();
  ShapedBuffer tuple_buffer = local_shaped_buffer->Release();

  // Extract some metadata we use to construct scoped buffers.
  const se::Platform* platform = tuple_buffer.platform();
  int device_ordinal = tuple_buffer.device_ordinal();

  ShapeTree<se::DeviceMemoryBase>& shape_tree = tuple_buffer.buffers();
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

StatusOr<XrtAllocationTuple*> DestructureXrtAllocationTuple(
    XrtAllocation* allocation, const string& session_target) {
  const Shape& tuple_shape = allocation->shape();

  if (!tuple_shape.IsTuple()) {
    return InvalidArgument(
        "Attemped to destructure a LocalShapedBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(tuple_shape));
  }

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  auto base_handle = tensorflow::ops::Placeholder(root, tensorflow::DT_INT64);
  auto shape_index = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
  auto subtuple = tensorflow::ops::XRTSubTuple(root, base_handle, shape_index);
  TF_RETURN_IF_ERROR(root.status());

  tensorflow::ClientSession session(root, session_target);
  tensorflow::ClientSession::FeedType inputs;
  std::vector<XrtAllocation*> results;
  for (int32 i = 0; i < ShapeUtil::TupleElementCount(tuple_shape); ++i) {
    inputs.clear();
    inputs.insert({base_handle, allocation->handle()});
    inputs.insert({shape_index, {i}});
    std::vector<tensorflow::Tensor> outputs;
    auto status = session.Run(inputs, {subtuple}, &outputs);
    if (!status.ok()) {
      // Clean up before returning non-ok status.
      for (int j = 0; j < results.size(); ++j) {
        delete results[j];
      }
      return status;
    }
    const int64 subtuple_handle = outputs[0].scalar<int64>()();
    const Shape& subtuple_shape =
        ShapeUtil::GetTupleElementShape(tuple_shape, i);
    results.push_back(
        new XrtAllocation(subtuple_handle, subtuple_shape, session_target));
  }
  return new XrtAllocationTuple(std::move(results));
}

}  // namespace swig
}  // namespace xla
