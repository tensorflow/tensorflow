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
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/cholesky.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace swig {

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

LocalClient::LocalClient(xla::LocalClient* client) : client_(client) {}

/* static */ StatusOr<LocalClient> LocalClient::Get(
    const string& platform_name) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(platform_name));
  if (platform->VisibleDeviceCount() <= 0) {
    return InvalidArgument("Platform %s has no visible devices.",
                           platform_name);
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(xla::LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));
  CHECK(client != nullptr);
  return LocalClient(client);
}

// Returns the number of devices known to the XLA client.
int LocalClient::DeviceCount() const { return client_->device_count(); }

Status LocalClient::TransferToInfeed(const Literal& literal,
                                     int device_ordinal) {
  VLOG(1) << "Infeeding literal to device " << device_ordinal
          << "; shape: " << literal.shape();
  return client_->TransferToInfeed(literal, device_ordinal);
}

StatusOr<Literal> LocalClient::TransferFromOutfeed(const Shape& shape,
                                                   int device_ordinal) {
  VLOG(1) << "Outfeeding literal from device " << device_ordinal
          << "; shape: " << shape;
  return client_->TransferFromOutfeed(&shape, device_ordinal);
}

/* static */
StatusOr<LocalShapedBuffer*> LocalShapedBuffer::FromLiteral(
    const Literal& argument, const absl::optional<Shape>& shape_with_layout,
    const LocalClient& client, int device_ordinal) {
  VLOG(1) << "Creating shaped buffer from literal on device ordinal: "
          << device_ordinal;
  auto literal_to_buffer = [&](const Literal& arg) {
    return client.client()->LiteralToShapedBuffer(
        arg, device_ordinal, client.client()->backend().memory_allocator());
  };

  StatusOr<ScopedShapedBuffer> buf = [&] {
    if (shape_with_layout) {
      Literal relaid = argument.Relayout(shape_with_layout.value());
      return literal_to_buffer(relaid);
    }
    return literal_to_buffer(argument);
  }();
  TF_RETURN_IF_ERROR(buf.status());
  return new LocalShapedBuffer(std::move(buf).ValueOrDie(), client.client());
}

LocalShapedBuffer::LocalShapedBuffer(ScopedShapedBuffer shaped_buffer,
                                     xla::LocalClient* client)
    : shaped_buffer_(std::move(shaped_buffer)), client_(client) {}

const ScopedShapedBuffer* LocalShapedBuffer::shaped_buffer() const {
  return &shaped_buffer_;
}

ShapedBuffer LocalShapedBuffer::Release() { return shaped_buffer_.release(); }

const Shape& LocalShapedBuffer::shape() const {
  return shaped_buffer()->on_device_shape();
}

StatusOr<Literal> LocalShapedBuffer::ToLiteral() const {
  return client_->ShapedBufferToLiteral(*shaped_buffer());
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

StatusOr<LocalShapedBufferTuple*> LocalShapedBuffer::DestructureTuple() {
  const Shape tuple_shape = shape();

  if (!tuple_shape.IsTuple()) {
    return InvalidArgument(
        "Attemped to destructure a LocalShapedBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(tuple_shape));
  }

  DeviceMemoryAllocator* allocator = shaped_buffer()->memory_allocator();
  ShapedBuffer tuple_buffer = Release();

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
        ScopedShapedBuffer(std::move(shaped_buffer), allocator), client_));
  }
  // Deallocate the root buffer.
  se::DeviceMemoryBase root_buffer = tuple_buffer.root_buffer();
  TF_RETURN_IF_ERROR(allocator->Deallocate(device_ordinal, root_buffer));
  return new LocalShapedBufferTuple(std::move(results));
}

LocalExecutable::LocalExecutable(
    std::unique_ptr<xla::LocalExecutable> executable,
    xla::DeviceAssignment device_assignment, xla::LocalClient* client)
    : executable_(std::move(executable)),
      device_assignment_(std::move(device_assignment)),
      client_(client) {}

std::vector<int> LocalExecutable::DeviceOrdinals() const {
  int num_replicas = device_assignment_.replica_count();
  std::vector<int> device_ordinals;
  device_ordinals.reserve(num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    device_ordinals.push_back(device_assignment_(i, 0));
  }
  return device_ordinals;
}

StatusOr<LocalShapedBuffer*> LocalExecutable::Execute(
    absl::Span<LocalShapedBuffer* const> argument_handles) {
  if (num_replicas() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d replicas using Execute()",
        num_replicas());
  }
  StatusOr<ScopedShapedBuffer> result_buffer_status;
  const int device_ordinal = device_assignment_(0, 0);
  VLOG(3) << "Replica 0 mapped to device ordinal for execution: "
          << device_ordinal;

  std::vector<const ShapedBuffer*> argument_buffers;
  argument_buffers.reserve(argument_handles.size());
  for (auto& handle : argument_handles) {
    argument_buffers.push_back(handle->shaped_buffer());
  }

  ExecutableRunOptions options;
  options.set_device_ordinal(device_ordinal);
  options.set_allocator(client_->backend().memory_allocator());
  options.set_intra_op_thread_pool(
      client_->backend().eigen_intra_op_thread_pool_device());
  options.set_device_assignment(&device_assignment_);

  result_buffer_status = executable_->Run(argument_buffers, options);

  if (!result_buffer_status.ok()) {
    return InternalError(
        "Failed running replica 0 (other replicas may have failed as well): "
        "%s.",
        result_buffer_status.status().ToString());
  }
  return new LocalShapedBuffer(std::move(result_buffer_status).ValueOrDie(),
                               client_);
}

StatusOr<LocalShapedBufferTuple*> LocalExecutable::ExecutePerReplica(
    absl::Span<const std::vector<LocalShapedBuffer*>> argument_handles) {
  const int num_devices = client_->device_count();

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

  std::vector<StatusOr<ScopedShapedBuffer>> results(num_replicas());
  auto execute = [this, &argument_handles, &results](int replica) {
    const int device_ordinal = device_assignment_(replica, 0);
    VLOG(3) << "Replica " << replica
            << " mapped to device ordinal for execution: " << device_ordinal;

    std::vector<const ShapedBuffer*> argument_buffers;
    argument_buffers.reserve(argument_handles[replica].size());
    for (auto& handle : argument_handles[replica]) {
      argument_buffers.push_back(handle->shaped_buffer());
    }

    ExecutableRunOptions options;
    options.set_device_ordinal(device_ordinal);
    options.set_allocator(client_->backend().memory_allocator());
    options.set_intra_op_thread_pool(
        client_->backend().eigen_intra_op_thread_pool_device());
    options.set_device_assignment(&device_assignment_);
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
        new LocalShapedBuffer(std::move(statusor).ValueOrDie(), client_);
  }

  return new LocalShapedBufferTuple(std::move(wrapped_results));
}

Computation::Computation(XlaComputation computation)
    : computation_(std::move(computation)) {}

StatusOr<LocalExecutable*> Computation::Compile(
    const std::vector<Shape>& argument_shapes,
    const ExecutableBuildOptions* build_options, const LocalClient& client) {
  std::vector<const Shape*> argument_shape_pointers;
  argument_shape_pointers.reserve(argument_shapes.size());
  for (auto& argument_shape : argument_shapes) {
    argument_shape_pointers.push_back(&argument_shape);
  }

  ExecutableBuildOptions options;
  if (build_options != nullptr) {
    options = *build_options;
  }
  TF_ASSIGN_OR_RETURN(
      auto local_executable,
      client.client()->Compile(computation_, argument_shape_pointers, options));
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      client.client()->backend().computation_placer()->AssignDevices(
          options.num_replicas(), /*computation_count=*/1));

  return new LocalExecutable(std::move(local_executable),
                             std::move(device_assignment), client.client());
}

const XlaComputation& Computation::computation() const { return computation_; }

string Computation::GetSerializedProto() const {
  string result;
  if (!computation_.proto().SerializeToString(&result)) {
    LOG(ERROR) << "Failed to serialize the HloModuleProto.";
    return "";
  }
  return result;
}

StatusOr<string> Computation::GetHloText() const {
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          computation_.proto(), GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation_.proto(), module_config));
  HloPrintOptions options;
  options = HloPrintOptions::ShortParsable();
  options.set_print_large_constants(false);
  return hlo_module->ToString(options);
}

StatusOr<string> Computation::GetHloDotGraph() const {
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          computation_.proto(), GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation_.proto(), module_config));
  hlo_graph_dumper::DotGraphOptions options;
  options.debug_options = &hlo_module->config().debug_options();
  return hlo_graph_dumper::HloComputationToDotGraph(
      *hlo_module->entry_computation(), options);
}

StatusOr<ProgramShape> Computation::GetProgramShape() const {
  return computation_.GetProgramShape();
}

StatusOr<Shape> Computation::GetReturnValueShape() const {
  TF_ASSIGN_OR_RETURN(ProgramShape shape, computation_.GetProgramShape());
  return std::move(*shape.mutable_result());
}

LocalOp::LocalOp(const XlaOp& op) : op_(op) {}

const XlaOp& LocalOp::op() const { return op_; }

ComputationBuilder::ComputationBuilder(const string& computation_name)
    : builder_(computation_name) {}

void ComputationBuilder::SetOpMetadata(const OpMetadata& metadata) {
  builder_.SetOpMetadata(metadata);
}

void ComputationBuilder::ClearOpMetadata() { builder_.ClearOpMetadata(); }

StatusOr<Computation*> ComputationBuilder::Build() {
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder_.Build());
  return new Computation(std::move(computation));
}

LocalOp ComputationBuilder::Parameter(int64 parameter_number,
                                      const Shape& shape, const string& name) {
  return xla::Parameter(&builder_, parameter_number, shape, name);
}

StatusOr<Computation*> ComputationBuilder::BuildWithRoot(const LocalOp& root) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation, builder_.Build(root.op()));
  return new Computation(std::move(computation));
}

StatusOr<Shape> ComputationBuilder::GetShape(const LocalOp& operand) {
  return builder_.GetShape(operand.op());
}

StatusOr<Shape> ComputationBuilder::GetReturnValueShape() {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape, builder_.GetProgramShape());
  return program_shape.result();
}

LocalOp ComputationBuilder::ReplicaId() { return xla::ReplicaId(&builder_); }

LocalOp ComputationBuilder::Infeed(const Shape& shape) {
  return xla::Infeed(&builder_, shape);
}

void ComputationBuilder::Outfeed(const LocalOp& operand, const Shape& shape,
                                 const string& outfeed_config) {
  xla::Outfeed(operand.op(), shape, outfeed_config);
}

LocalOp ComputationBuilder::ConstantLiteral(const Literal& literal) {
  return xla::ConstantLiteral(&builder_, literal);
}

LocalOp ComputationBuilder::Iota(PrimitiveType element_type, int64 size) {
  return xla::Iota(&builder_, element_type, size);
}

LocalOp ComputationBuilder::BroadcastedIota(const Shape& shape,
                                            int64 dimension) {
  return xla::Iota(&builder_, shape, dimension);
}

LocalOp ComputationBuilder::Broadcast(const LocalOp& operand,
                                      absl::Span<const int64> broadcast_sizes) {
  return xla::Broadcast(operand.op(), broadcast_sizes);
}

LocalOp ComputationBuilder::BroadcastInDim(
    const LocalOp& operand, absl::Span<const int64> out_dim_sizes,
    absl::Span<const int64> broadcast_dimensions) {
  return xla::BroadcastInDim(operand.op(), out_dim_sizes, broadcast_dimensions);
}

LocalOp ComputationBuilder::Pad(const LocalOp& operand,
                                const LocalOp& padding_value,
                                const PaddingConfig& padding_config) {
  return xla::Pad(operand.op(), padding_value.op(), padding_config);
}

LocalOp ComputationBuilder::Reshape(const LocalOp& operand,
                                    absl::Span<const int64> dimensions,
                                    absl::Span<const int64> new_sizes) {
  return xla::Reshape(operand.op(), dimensions, new_sizes);
}

LocalOp ComputationBuilder::Collapse(const LocalOp& operand,
                                     absl::Span<const int64> dimensions) {
  return xla::Collapse(operand.op(), dimensions);
}

LocalOp ComputationBuilder::AllToAll(
    const LocalOp& operand, int64 split_dimension, int64 concat_dimension,
    int64 split_count, absl::Span<const ReplicaGroup> replica_groups) {
  std::vector<ReplicaGroup> rg(replica_groups.size());
  for (int i = 0; i < replica_groups.size(); ++i) {
    rg.push_back(replica_groups[i]);
  }
  return xla::AllToAll(operand.op(), split_dimension, concat_dimension,
                       split_count, rg);
}

LocalOp ComputationBuilder::CrossReplicaSum(
    const LocalOp& operand, absl::Span<const ReplicaGroup> replica_groups) {
  return xla::CrossReplicaSum(operand.op(), replica_groups);
}

LocalOp ComputationBuilder::Slice(const LocalOp& operand,
                                  absl::Span<const int64> start_indices,
                                  absl::Span<const int64> limit_indices,
                                  absl::Span<const int64> strides) {
  return xla::Slice(operand.op(), start_indices, limit_indices, strides);
}

LocalOp ComputationBuilder::SliceInDim(const LocalOp& operand,
                                       int64 start_index, int64 limit_index,
                                       int64 stride, int64 dimno) {
  return xla::SliceInDim(operand.op(), start_index, limit_index, stride, dimno);
}

LocalOp ComputationBuilder::DynamicSlice(const LocalOp& operand,
                                         const LocalOp& start_indices,
                                         absl::Span<const int64> slice_sizes) {
  return xla::DynamicSlice(operand.op(), start_indices.op(), slice_sizes);
}

LocalOp ComputationBuilder::DynamicUpdateSlice(const LocalOp& operand,
                                               const LocalOp& update,
                                               const LocalOp& start_indices) {
  return xla::DynamicUpdateSlice(operand.op(), update.op(), start_indices.op());
}

LocalOp ComputationBuilder::ConcatInDim(absl::Span<const LocalOp> operands,
                                        int64 dimension) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }
  return xla::ConcatInDim(&builder_, xla_ops, dimension);
}

LocalOp ComputationBuilder::SelectAndScatterWithGeneralPadding(
    const LocalOp& operand, const Computation& select,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding, const LocalOp& source,
    const LocalOp& init_value, const Computation& scatter) {
  return xla::SelectAndScatterWithGeneralPadding(
      operand.op(), select.computation(), window_dimensions, window_strides,
      padding, source.op(), init_value.op(), scatter.computation());
}

LocalOp ComputationBuilder::Tuple(absl::Span<const LocalOp> elements) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(elements.size());
  for (const auto& op : elements) {
    xla_ops.push_back(op.op());
  }

  return xla::Tuple(&builder_, xla_ops);
}

LocalOp ComputationBuilder::GetTupleElement(const LocalOp& tuple_data,
                                            int64 index) {
  return xla::GetTupleElement(tuple_data.op(), index);
}

LocalOp ComputationBuilder::Dot(const LocalOp& lhs, const LocalOp& rhs) {
  return xla::Dot(lhs.op(), rhs.op());
}

LocalOp ComputationBuilder::DotGeneral(
    const LocalOp& lhs, const LocalOp& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  return xla::DotGeneral(lhs.op(), rhs.op(), dimension_numbers);
}

LocalOp ComputationBuilder::ConvGeneralDilated(
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

LocalOp ComputationBuilder::ConvertElementType(const LocalOp& operand,
                                               PrimitiveType new_element_type) {
  return xla::ConvertElementType(operand.op(), new_element_type);
}

LocalOp ComputationBuilder::BitcastConvertType(const LocalOp& operand,
                                               PrimitiveType new_element_type) {
  return xla::BitcastConvertType(operand.op(), new_element_type);
}

LocalOp ComputationBuilder::Call(const Computation& local_computation,
                                 absl::Span<const LocalOp> operands) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }
  return xla::Call(&builder_, local_computation.computation(), xla_ops);
}

LocalOp ComputationBuilder::CustomCall(
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

LocalOp ComputationBuilder::Transpose(const LocalOp& operand,
                                      absl::Span<const int64> permutation) {
  return xla::Transpose(operand.op(), permutation);
}

LocalOp ComputationBuilder::Rev(const LocalOp& operand,
                                absl::Span<const int64> dimensions) {
  return xla::Rev(operand.op(), dimensions);
}

LocalOp ComputationBuilder::Map(absl::Span<const LocalOp> operands,
                                const Computation& local_computation,
                                absl::Span<const int64> dimensions) {
  std::vector<XlaOp> xla_ops;
  xla_ops.reserve(operands.size());
  for (const auto& op : operands) {
    xla_ops.push_back(op.op());
  }

  return xla::Map(&builder_, xla_ops, local_computation.computation(),
                  dimensions);
}

LocalOp ComputationBuilder::Reduce(
    const LocalOp& operand, const LocalOp& init_value,
    const Computation& local_computation,
    absl::Span<const int64> dimensions_to_reduce) {
  return xla::Reduce(operand.op(), init_value.op(),
                     local_computation.computation(), dimensions_to_reduce);
}

LocalOp ComputationBuilder::ReduceWindowWithGeneralPadding(
    const LocalOp& operand, const LocalOp& init_value,
    const Computation& local_computation,
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

LocalOp ComputationBuilder::RngNormal(const LocalOp& mu, const LocalOp& sigma,
                                      const Shape& shape) {
  return xla::RngNormal(mu.op(), sigma.op(), shape);
}

LocalOp ComputationBuilder::RngUniform(const LocalOp& a, const LocalOp& b,
                                       const Shape& shape) {
  return xla::RngUniform(a.op(), b.op(), shape);
}

LocalOp ComputationBuilder::While(const Computation& condition,
                                  const Computation& body,
                                  const LocalOp& init) {
  return xla::While(condition.computation(), body.computation(), init.op());
}

LocalOp ComputationBuilder::Conditional(const LocalOp& predicate,
                                        const LocalOp& true_operand,
                                        const Computation& true_computation,
                                        const LocalOp& false_operand,
                                        const Computation& false_computation) {
  return xla::Conditional(predicate.op(), true_operand.op(),
                          true_computation.computation(), false_operand.op(),
                          false_computation.computation());
}

StatusOr<bool> ComputationBuilder::IsConstant(const LocalOp& operand) {
  return builder_.IsConstant(operand.op());
}

LocalOp ComputationBuilder::Sort(const LocalOp& operand, int64 dimension) {
  return xla::Sort(operand.op(), {}, dimension);
}

LocalOp ComputationBuilder::SortKeyVal(const LocalOp& keys,
                                       const LocalOp& values, int64 dimension) {
  return xla::Sort(keys.op(), {values.op()}, dimension);
}

LocalOp ComputationBuilder::Cholesky(const LocalOp& a) {
  return xla::Cholesky(a.op());
}

LocalOp ComputationBuilder::QR(const LocalOp& a, bool full_matrices) {
  XlaBuilder* builder = a.op().builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto qr, xla::QRDecomposition(a.op(), full_matrices));
    return xla::Tuple(builder, {qr.q, qr.r});
  });
}

LocalOp ComputationBuilder::TriangularSolve(const LocalOp& a, const LocalOp& b,
                                            bool left_side, bool lower,
                                            bool unit_diagonal,
                                            int transpose_a) {
  return xla::TriangularSolve(
      a.op(), b.op(), left_side, lower, unit_diagonal,
      xla::TriangularSolveOptions::Transpose(transpose_a));
}

LocalOp ComputationBuilder::Gather(
    const LocalOp& input, const LocalOp& start_indices,
    const GatherDimensionNumbers& dimension_numbers,
    absl::Span<const int64> slice_sizes) {
  return xla::Gather(input.op(), start_indices.op(), dimension_numbers,
                     slice_sizes);
}

LocalOp ComputationBuilder::Scatter(
    const LocalOp& input, const LocalOp& scatter_indices,
    const LocalOp& updates, const Computation& update_computation,
    const ScatterDimensionNumbers& dimension_numbers) {
  return xla::Scatter(input.op(), scatter_indices.op(), updates.op(),
                      update_computation.computation(), dimension_numbers);
}

StatusOr<Computation*> ComputationBuilder::BuildConstantSubGraph(
    const LocalOp& operand) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation,
                      builder_.BuildConstantSubGraph(operand.op()));
  return new Computation(std::move(computation));
}

#define _FORWARD(method_name, return_sig, args_sig, args) \
  return_sig ComputationBuilder::method_name args_sig {   \
    return xla::method_name args;                         \
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
_FORWARD_UNOP(Clz)
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

void DeleteLocalExecutable(LocalExecutable* computation) { delete computation; }

void DeleteComputation(Computation* computation) { delete computation; }

}  // namespace swig
}  // namespace xla
