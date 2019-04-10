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

#include "tensorflow/compiler/xla/python/local_client.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace xla_python {

namespace py = pybind11;

// Registers a 'fn_capsule' as a CPU custom call target.
// 'fn_capsule' is a void* pointer encapsulated in a PyCapsule object, with name
// "xla._CPU_CUSTOM_CALL_TARGET".
Status RegisterCpuCustomCallTarget(const std::string& fn_name,
                                   py::capsule capsule) {
  static const char* const kName = "xla._CPU_CUSTOM_CALL_TARGET";
  if (absl::string_view(capsule.name()) != kName) {
    return InvalidArgument(
        "Argument to RegisterCpuCustomCallTargetRegistry was not a "
        "xla._CPU_CUSTOM_CALL_TARGET capsule.");
  }
  cpu::CustomCallTargetRegistry::Global()->Register(
      std::string(fn_name.begin(), fn_name.end()), static_cast<void*>(capsule));
  return Status::OK();
}

StatusOr<LocalClient*> GetLocalClient(const std::string& platform_name) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(platform_name));
  if (platform->VisibleDeviceCount() <= 0) {
    return InvalidArgument("Platform %s has no visible devices.",
                           platform_name);
  }
  LocalClientOptions options;
  options.set_platform(platform);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

/* static */
StatusOr<LocalShapedBuffer> LocalShapedBuffer::FromPython(
    const py::object& argument, LocalClient* client, int device_ordinal) {
  VLOG(1) << "Creating shaped buffer from literal on device ordinal: "
          << device_ordinal;
  TF_ASSIGN_OR_RETURN(PythonBufferTree tree, GetPythonBufferTree(argument));

  // We are done manipulating Python objects; release the GIL.
  py::gil_scoped_release gil_release;
  DeviceMemoryAllocator* allocator = client->backend().memory_allocator();
  TransferManager* transfer_manager = client->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          tree.shape, allocator, device_ordinal));
  TF_ASSIGN_OR_RETURN(auto stream,
                      client->mutable_backend()->BorrowStream(device_ordinal));
  TF_RETURN_IF_ERROR(
      transfer_manager->WriteTupleIndexTables(stream.get(), buffer));

  auto it = tree.leaves.begin();
  for (const ShapeUtil::IndexedShape& indexed_shape :
       ShapeUtil::GetLeafShapes(tree.shape)) {
    TF_RET_CHECK(it != tree.leaves.end());
    ShapedBuffer leaf(
        indexed_shape.shape,
        transfer_manager->HostShapeToDeviceShape(indexed_shape.shape),
        client->platform(), device_ordinal);
    leaf.buffers().CopySubtreeFrom(buffer.buffers(), indexed_shape.index, {});
    TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDeviceAsync(
        stream.get(), *it, leaf));
    ++it;
  }
  stream->BlockHostUntilDone();
  return LocalShapedBuffer(std::move(buffer), client);
}

LocalShapedBuffer::LocalShapedBuffer(ScopedShapedBuffer shaped_buffer,
                                     LocalClient* client)
    : shaped_buffer_(std::move(shaped_buffer)), client_(client) {}

const ScopedShapedBuffer* LocalShapedBuffer::shaped_buffer() const {
  return &shaped_buffer_.value();
}

ScopedShapedBuffer LocalShapedBuffer::Release() {
  ScopedShapedBuffer result = std::move(*shaped_buffer_);
  shaped_buffer_ = absl::nullopt;
  return result;
}

const Shape& LocalShapedBuffer::shape() const {
  return shaped_buffer()->on_device_shape();
}

StatusOr<py::object> LocalShapedBuffer::ToPython() const {
  auto literal = absl::make_unique<Literal>();
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(*literal,
                        client_->ShapedBufferToLiteral(*shaped_buffer()));
  }
  return LiteralToPython(std::move(literal));
}

StatusOr<std::vector<LocalShapedBuffer>> LocalShapedBuffer::DestructureTuple() {
  const Shape tuple_shape = shape();

  if (!tuple_shape.IsTuple()) {
    return InvalidArgument(
        "Attemped to destructure a LocalShapedBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(tuple_shape));
  }

  DeviceMemoryAllocator* allocator = shaped_buffer()->memory_allocator();
  ScopedShapedBuffer tuple_buffer = Release();

  // Extract some metadata we use to construct scoped buffers.
  const se::Platform* platform = tuple_buffer.platform();
  int device_ordinal = tuple_buffer.device_ordinal();

  ShapeTree<se::DeviceMemoryBase>& shape_tree = tuple_buffer.buffers();
  std::vector<LocalShapedBuffer> results;
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
    results.push_back(LocalShapedBuffer(
        ScopedShapedBuffer(std::move(shaped_buffer), allocator), client_));
  }
  return results;
}

LocalExecutableWrapper::LocalExecutableWrapper(
    std::unique_ptr<LocalExecutable> executable,
    DeviceAssignment device_assignment, LocalClient* client)
    : executable_(std::move(executable)),
      device_assignment_(std::move(device_assignment)),
      client_(client) {}

std::vector<int> LocalExecutableWrapper::DeviceOrdinals() const {
  int num_replicas = device_assignment_.replica_count();
  std::vector<int> device_ordinals;
  device_ordinals.reserve(num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    device_ordinals.push_back(device_assignment_(i, 0));
  }
  return device_ordinals;
}

StatusOr<LocalShapedBuffer> LocalExecutableWrapper::Execute(
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
  return LocalShapedBuffer(std::move(result_buffer_status).ValueOrDie(),
                           client_);
}

StatusOr<std::vector<LocalShapedBuffer>>
LocalExecutableWrapper::ExecutePerReplica(
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

  std::vector<LocalShapedBuffer> wrapped_results(num_replicas());
  for (int replica = 0; replica < num_replicas(); ++replica) {
    auto& statusor = results[replica];
    if (!statusor.ok()) {
      return InternalError(
          "Failed running replica %d (other replicas may have failed as well): "
          "%s.",
          replica, statusor.status().ToString());
    }
    wrapped_results[replica] =
        LocalShapedBuffer(std::move(statusor).ValueOrDie(), client_);
  }
  return wrapped_results;
}

StatusOr<py::bytes> GetComputationSerializedProto(
    const XlaComputation& computation) {
  std::string result;
  if (!computation.proto().SerializeToString(&result)) {
    return Unknown("Failed to serialize the HloModuleProto.");
  }
  return py::bytes(result);
}

StatusOr<std::string> GetComputationHloText(const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          computation.proto(), GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation.proto(), module_config));
  HloPrintOptions options;
  options = HloPrintOptions::ShortParsable();
  options.set_print_large_constants(false);
  return hlo_module->ToString(options);
}

StatusOr<std::string> GetComputationHloDotGraph(
    const XlaComputation& computation) {
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          computation.proto(), GetDebugOptionsFromFlags()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProto(computation.proto(), module_config));
  return RenderGraph(*hlo_module->entry_computation(), /*label=*/"",
                     hlo_module->config().debug_options(),
                     RenderedGraphFormat::kDot);
}

/*static*/ StatusOr<std::unique_ptr<LocalExecutableWrapper>>
LocalExecutableWrapper::Compile(const XlaComputation& computation,
                                const std::vector<Shape>& argument_shapes,
                                const ExecutableBuildOptions* build_options,
                                LocalClient* client) {
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
      client->Compile(computation, argument_shape_pointers, options));
  TF_ASSIGN_OR_RETURN(DeviceAssignment device_assignment,
                      client->backend().computation_placer()->AssignDevices(
                          options.num_replicas(), /*computation_count=*/1));

  return absl::make_unique<LocalExecutableWrapper>(
      std::move(local_executable), std::move(device_assignment), client);
}

}  // namespace xla_python
}  // namespace xla
