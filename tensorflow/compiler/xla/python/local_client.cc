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

// Implementation notes:
//
// Asynchronous execution:
// -----------------------
//
// If 'asynchronous' is set when constructing the client, computations and
// host-to-device transfers do not block the host waiting for the operation to
// complete but instead return control to the host immediately. This allows
// Python logic to overlap with device-side computation.
//
// For a good user experience, we must be careful only to enqueue operations
// that are unlikely to fail; as a rule error checking must be done eagerly
// before returning control to the client.
//
// Multi-stream execution:
// -----------------------
//
// On certain platforms (e.g., TPU), we use a multistream execution design,
// where different Streams are used for host-to-device transfers,
// device-to-host transfers, and compute. This allows us to overlap transfers on
// and off the device with computation.
//
// Synchronization between streams occurs via BufferDefinitionEvents that
// describe when the contents of a logical buffer are known to be valid on
// a particular stream.
//
// Synchronous vs asynchronous deallocation:
// -----------------------------------------
//
// In asynchronous deallocation mode (currently only enabled on TPU), the client
// need only keep buffers alive from its perspective until all operations that
// touch those buffers have been enqueued.
// The allocator and lower-level runtime is responsible for keeping buffers
// alive (if that is needed) from the perspective of the device until any
// device-side work actually completes. The client's use of the device allocator
// thereby corresponds to a view of the tail of the compute stream instead of
// its head.
//
// In synchronous deallocation mode the client is responsible for keeping
// buffers alive until all device-side activity that consumes those buffers has
// ceased. This is the case for CPU since HostExecutor performs allocation
// and deallocation eagerly. In this mode, the client's use of the device
// allocator is logically synchronized to the head of the compute stream, not
// the tail.

#include "tensorflow/compiler/xla/python/local_client.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/python/shared_device_buffer.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace xla {

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
  CustomCallTargetRegistry::Global()->Register(
      fn_name, static_cast<void*>(capsule), "Host");
  return Status::OK();
}

Device::Device(se::StreamExecutor* executor, bool use_multiple_streams,
               bool synchronous_deallocation, bool asynchronous)
    : use_multiple_streams_(use_multiple_streams),
      synchronous_deallocation_(synchronous_deallocation),
      asynchronous_(asynchronous) {
  compute_stream_ = std::make_shared<se::Stream>(executor);
  compute_stream_->Init();
  if (use_multiple_streams) {
    host_to_device_stream_ = std::make_shared<se::Stream>(executor);
    device_to_host_stream_ = std::make_shared<se::Stream>(executor);
    callback_stream_ = std::make_shared<se::Stream>(executor);
    host_to_device_stream_->Init();
    device_to_host_stream_->Init();
    callback_stream_->Init();
  } else {
    callback_stream_ = host_to_device_stream_ = device_to_host_stream_ =
        compute_stream_;
  }
  worker_thread_ = absl::make_unique<WorkerThread>(tensorflow::Env::Default(),
                                                   "py_xla_execute");
}

Device::~Device() {
  Status status = SynchronizeAllActivity();
  if (!status.ok()) {
    LOG(ERROR) << "Error when closing device: " << status;
  }
}

Status Device::SynchronizeAllActivity() {
  Status status;
  // TODO(phawkins): in theory the call to SynchronizeAllActivity below should
  // suffice. However on the Host platform SynchronizeAllActivity is a dummy
  // implementation that doesn't actually block. To make sure activity has
  // stopped, also block on the compute stream. If SynchronizeAllActivity is
  // fixed, we could remove the BlockHostUntilDone call.
  status.Update(compute_stream_->BlockHostUntilDone());
  bool ok = compute_stream_->parent()->SynchronizeAllActivity();
  if (!ok) {
    status.Update(Unknown("SynchronizeAllActivity failed."));
  }
  return status;
}

void Device::ThenExecuteOnWorkerThread(se::Stream* stream,
                                       std::function<void()> callback) const {
  stream->ThenDoHostCallback(
      [this, callback]() { worker_thread_->Schedule(std::move(callback)); });
}

static StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(
    se::Platform* platform, LocalClient* client, double memory_fraction,
    bool preallocate) {
  CHECK_GT(client->backend().device_count(), 0);
  std::vector<std::unique_ptr<tensorflow::Allocator>> allocators;
  for (se::StreamExecutor* executor : client->backend().stream_executors()) {
    int device_ordinal = executor->device_ordinal();
    auto sub_allocator = absl::make_unique<tensorflow::GPUMemAllocator>(
        executor, tensorflow::PlatformGpuId(device_ordinal),
        /*use_unified_memory=*/false,
        /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
        /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

    int64 free_memory;
    int64 total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
    }
    size_t allocator_memory = free_memory * memory_fraction;
    if (preallocate) {
      LOG(INFO) << "XLA backend allocating " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    } else {
      LOG(INFO) << "XLA backend will use up to " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for BFCAllocator.";
    }
    auto gpu_bfc_allocator = absl::make_unique<tensorflow::BFCAllocator>(
        sub_allocator.release(), allocator_memory,
        /*allow_growth=*/!preallocate,
        absl::StrCat("GPU_", device_ordinal, "_bfc"));
    allocators.emplace_back(std::move(gpu_bfc_allocator));
  }
  return absl::make_unique<se::MultiDeviceAdapter>(platform,
                                                   std::move(allocators));
}

StatusOr<std::shared_ptr<PyLocalClient>> PyLocalClient::Get(
    const std::string& platform_name, const std::string& xla_platform_name,
    bool asynchronous, const AllocatorConfig& allocator_config) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(xla_platform_name));
  if (platform->VisibleDeviceCount() <= 0) {
    return InvalidArgument("Platform %s (%s) has no visible devices.",
                           platform_name, xla_platform_name);
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));
  std::unique_ptr<se::DeviceMemoryAllocator> allocator;
  if (allocator_config.kind == AllocatorConfig::Kind::kBFC ||
      (platform_name == "gpu" &&
       allocator_config.kind == AllocatorConfig::Kind::kDefault)) {
    if (platform_name != "gpu") {
      return Unimplemented("BFCAllocator only available for GPU.");
    }
    TF_ASSIGN_OR_RETURN(
        auto bfc_allocator,
        CreateBFCAllocator(platform, client, allocator_config.memory_fraction,
                           allocator_config.preallocate));
    allocator = std::move(bfc_allocator);
  }
  return std::make_shared<PyLocalClient>(platform_name, client,
                                         std::move(allocator), asynchronous);
}

PyLocalClient::PyLocalClient(
    std::string platform_name, LocalClient* client,
    std::unique_ptr<se::DeviceMemoryAllocator> allocator, bool asynchronous)
    : platform_name_(std::move(platform_name)),
      client_(client),
      owned_allocator_(std::move(allocator)),
      h2d_transfer_pool_(tensorflow::Env::Default(), "py_xla_h2d_transfer",
                         client->device_count()) {
  if (owned_allocator_ != nullptr) {
    allocator_ = owned_allocator_.get();
  } else {
    allocator_ = client_->backend().memory_allocator();
  }
  devices_.reserve(client->device_count());
  bool use_multiple_streams = (platform_name_ != "cpu");
  bool synchronous_deallocation = !use_multiple_streams;
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor =
        client_->backend().stream_executor(i).ValueOrDie();
    devices_.push_back(absl::make_unique<Device>(executor, use_multiple_streams,
                                                 synchronous_deallocation,
                                                 asynchronous));
  }
}

Status PyLocalClient::TransferToInfeed(const LiteralSlice& literal,
                                       int device_ordinal) {
  py_ref_manager().CollectGarbage();
  py::gil_scoped_release gil_release;
  return client_->TransferToInfeedLocal(literal, device_ordinal);
}

StatusOr<pybind11::object> PyLocalClient::TransferFromOutfeed(
    const Shape& shape, int device_ordinal) {
  py_ref_manager().CollectGarbage();
  Literal literal;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(
        literal, client_->TransferFromOutfeedLocal(shape, device_ordinal));
  }
  return LiteralToPython(std::make_shared<Literal>(std::move(literal)));
}

static StatusOr<std::unique_ptr<PyLocalBuffer>> TransferHostToDeviceAsync(
    const PythonBufferTree& tree, int device_ordinal,
    std::shared_ptr<PyLocalClient> client, const Device& device) {
  se::DeviceMemoryAllocator* allocator = client->allocator();
  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(
      Shape shape, transfer_manager->ChooseCompactLayoutForShape(tree.shape));
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          shape, allocator, device_ordinal));
  TF_RETURN_IF_ERROR(transfer_manager->WriteTupleIndexTablesAsync(
      device.host_to_device_stream(), buffer));

  auto it = tree.leaves.begin();
  for (const ShapeUtil::IndexedShape& indexed_shape :
       ShapeUtil::GetLeafShapes(shape)) {
    TF_RET_CHECK(it != tree.leaves.end());
    ShapedBuffer leaf(
        indexed_shape.shape,
        transfer_manager->HostShapeToDeviceShape(indexed_shape.shape),
        client->client()->platform(), device_ordinal);
    leaf.buffers().CopySubtreeFrom(buffer.buffers(), indexed_shape.index, {});
    if (device.use_multiple_streams() &&
        !transfer_manager->CanShapedBufferBeAccessedNow(
            device.host_to_device_stream()->parent(), leaf)) {
      device.host_to_device_stream()->ThenWaitFor(device.compute_stream());
    }
    TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDeviceAsync(
        device.host_to_device_stream(), *it, leaf));
    ++it;
  }
  std::shared_ptr<BufferDefinitionEvent> definition_event;
  if (device.use_multiple_streams()) {
    TF_ASSIGN_OR_RETURN(definition_event,
                        BufferDefinitionEvent::Create(
                            device.host_to_device_stream()->parent()));
    definition_event->RecordOnStream(device.host_to_device_stream());
  }
  std::shared_ptr<PySharedDeviceBuffer> device_buffer =
      PySharedDeviceBuffer::FromScopedShapedBuffer(std::move(buffer),
                                                   definition_event);
  if (device.synchronous_deallocation()) {
    device.ThenReleaseOnWorkerThread(device.host_to_device_stream(),
                                     device_buffer);
  }
  return absl::make_unique<PyLocalBuffer>(shape, std::move(device_buffer),
                                          std::move(client));
}

/* static */
StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::FromPython(
    const py::object& argument, std::shared_ptr<PyLocalClient> client,
    int device_ordinal) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::FromPython");
  TF_ASSIGN_OR_RETURN(PythonBufferTree tree, GetPythonBufferTree(argument));

  client->py_ref_manager().CollectGarbage();

  // Take a reference to the buffer to ensure that the inputs in host memory
  // remain live until the transfer is complete.
  auto py_buffer_ref =
      client->py_ref_manager().ManageReferences(absl::MakeSpan(tree.arrays));

  // We are done manipulating Python objects; release the GIL.
  py::gil_scoped_release gil_release;
  VLOG(1) << "PyLocalBuffer::FromPython: shape: " << tree.shape.ToString()
          << " device ordinal: " << device_ordinal;

  const Device& device = client->device(device_ordinal);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PyLocalBuffer> buffer,
                      TransferHostToDeviceAsync(tree, device_ordinal,
                                                std::move(client), device));

  device.ThenRelease(device.host_to_device_stream(), std::move(py_buffer_ref));
  return buffer;
}

/*static */ StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalBuffer::FromPythonValues(
    const std::vector<std::pair<py::object, int>>& arguments,
    std::shared_ptr<PyLocalClient> client) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::FromPythonValues");
  int num_arguments = static_cast<int>(arguments.size());
  std::vector<std::unique_ptr<PyLocalBuffer>> outputs(num_arguments);
  if (num_arguments == 0) {
    return outputs;
  }

  struct H2DTransfer {
    PythonBufferTree tree;
    StatusOr<std::unique_ptr<PyLocalBuffer>> buffer;
    PythonRefManager::ManagedPyObjects py_buffer_refs;
  };

  std::vector<H2DTransfer> transfers(num_arguments);
  for (int i = 0; i < num_arguments; ++i) {
    TF_ASSIGN_OR_RETURN(transfers[i].tree,
                        GetPythonBufferTree(arguments[i].first));
    transfers[i].py_buffer_refs = client->py_ref_manager().ManageReferences(
        absl::MakeSpan(transfers[i].tree.arrays));
  }
  client->py_ref_manager().CollectGarbage();
  // We are done manipulating Python objects; release the GIL.
  py::gil_scoped_release gil_release;

  auto transfer_h2d = [&](int i) -> StatusOr<std::unique_ptr<PyLocalBuffer>> {
    int device_ordinal = arguments[i].second;
    return TransferHostToDeviceAsync(transfers[i].tree, device_ordinal, client,
                                     client->device(device_ordinal));
  };

  // We perform the transfers on a thread pool in case XLA needs to do any
  // host-side preprocessing of the input data.
  if (num_arguments == 1) {
    transfers[0].buffer = transfer_h2d(0);
  } else {
    absl::BlockingCounter counter(num_arguments);
    for (int i = 0; i < num_arguments; ++i) {
      client->h2d_transfer_pool()->Schedule([&, i]() {
        transfers[i].buffer = transfer_h2d(i);
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }

  // Release our references once the transfers have completed.
  for (int i = 0; i < num_arguments; ++i) {
    int device_ordinal = arguments[i].second;
    const Device& device = client->device(device_ordinal);
    device.ThenRelease(device.host_to_device_stream(),
                       std::move(transfers[i].py_buffer_refs));
  }

  for (int i = 0; i < num_arguments; ++i) {
    TF_ASSIGN_OR_RETURN(outputs[i], std::move(transfers[i].buffer));
  }
  return outputs;
}

/* static */ StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::MakeTuple(
    const std::vector<PyLocalBuffer*> buffers,
    std::shared_ptr<PyLocalClient> client, int device_ordinal) {
  std::vector<xla::Shape> host_shapes;
  std::vector<std::shared_ptr<PySharedDeviceBuffer>> device_buffers;
  host_shapes.reserve(buffers.size());
  device_buffers.reserve(buffers.size());
  for (const PyLocalBuffer* buffer : buffers) {
    TF_RET_CHECK(buffer->device_ordinal() == device_ordinal);
    std::shared_ptr<PySharedDeviceBuffer> device_buffer =
        buffer->DeviceBuffer();
    if (!device_buffer) {
      return InvalidArgument(
          "Invalid buffer passed to MakeTuple() as argument %d.",
          device_buffers.size());
    }
    host_shapes.push_back(buffer->on_host_shape());
    device_buffers.push_back(std::move(device_buffer));
  }
  se::DeviceMemoryAllocator* allocator = client->allocator();
  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  const Device& device = client->device(device_ordinal);
  std::shared_ptr<BufferDefinitionEvent> definition_event;
  if (device.use_multiple_streams()) {
    TF_ASSIGN_OR_RETURN(definition_event,
                        BufferDefinitionEvent::Create(
                            device.host_to_device_stream()->parent()));
  }
  TF_ASSIGN_OR_RETURN(std::shared_ptr<PySharedDeviceBuffer> tuple_buffer,
                      PySharedDeviceBuffer::MakeTuple(
                          device_buffers, transfer_manager, allocator,
                          device_ordinal, definition_event));
  auto buffer = absl::make_unique<PyLocalBuffer>(
      ShapeUtil::MakeTupleShape(host_shapes), tuple_buffer, std::move(client));

  // TODO(phawkins): extend TransferManager so we do not need to form a full
  // ShapedBuffer just to write the root tuple index table.
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, buffer->AsShapedBuffer());
  if (device.use_multiple_streams() &&
      !transfer_manager->CanShapedBufferBeAccessedNow(
          device.host_to_device_stream()->parent(), shaped_buffer)) {
    // Wait for the compute stream so that memory allocations are synchronized.
    device.host_to_device_stream()->ThenWaitFor(device.compute_stream());
  }
  TF_RETURN_IF_ERROR(transfer_manager->WriteRootTupleIndexTable(
      device.host_to_device_stream(), shaped_buffer));
  if (definition_event) {
    definition_event->RecordOnStream(device.host_to_device_stream());
  }

  if (device.synchronous_deallocation()) {
    device.ThenReleaseOnWorkerThread(device.host_to_device_stream(),
                                     std::move(tuple_buffer));
  }
  return buffer;
}

PyLocalBuffer::PyLocalBuffer(
    Shape on_host_shape, std::shared_ptr<PySharedDeviceBuffer> device_buffer,
    std::shared_ptr<PyLocalClient> client)
    : client_(std::move(client)),
      on_host_shape_(std::move(on_host_shape)),
      device_ordinal_(device_buffer->device_ordinal()),
      device_buffer_(std::move(device_buffer)) {}

void PyLocalBuffer::Delete() {
  absl::MutexLock lock(&mu_);
  device_buffer_ = nullptr;
  host_value_ = nullptr;
}

Status PyLocalBuffer::CopyToHostAsync() {
  std::shared_ptr<PySharedDeviceBuffer> device_buffer;
  std::shared_ptr<HostValue> host_value;
  {
    absl::MutexLock lock(&mu_);
    if (!device_buffer_) {
      return InvalidArgument("CopyToHostAsync() called on invalid buffer.");
    }
    device_buffer = device_buffer_;

    if (host_value_) {
      // The host value has already been requested or is available.
      return Status::OK();
    }
    host_value = host_value_ = std::make_shared<HostValue>();
  }
  se::Stream* stream = client_->device(device_ordinal_).device_to_host_stream();
  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  host_value->value = std::make_shared<Literal>(on_host_shape_);
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, AsShapedBuffer());
  client_->client()->backend().transfer_manager()->TransferLiteralFromDevice(
      stream, shaped_buffer, *host_value->value,
      [host_value](Status done_status) {
        host_value->status = done_status;
        host_value->ready.Notify();
      });
  return Status::OK();
}

StatusOr<py::object> PyLocalBuffer::ToPython() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::ToPython");
  std::shared_ptr<PySharedDeviceBuffer> device_buffer = DeviceBuffer();
  if (!device_buffer) {
    return InvalidArgument("ToPython() called on invalid buffer.");
  }

  client_->py_ref_manager().CollectGarbage();
  std::shared_ptr<Literal> literal;
  {
    py::gil_scoped_release gil_release;
    TF_RETURN_IF_ERROR(CopyToHostAsync());
    std::shared_ptr<HostValue> host_value;
    {
      absl::MutexLock lock(&mu_);
      host_value = host_value_;
    }
    host_value->ready.WaitForNotification();
    TF_RETURN_IF_ERROR(host_value->status);
    literal = host_value->value;
  }

  return LiteralToPython(std::move(literal));
}

std::shared_ptr<PySharedDeviceBuffer> PyLocalBuffer::DeviceBuffer() const {
  absl::MutexLock lock(&mu_);
  return device_buffer_;
}

StatusOr<ShapedBuffer> PyLocalBuffer::AsShapedBuffer() const {
  absl::MutexLock lock(&mu_);
  if (!device_buffer_) {
    return InvalidArgument(
        "Attempted to fetch value of invalid/deleted buffer.");
  }
  return device_buffer_->AsShapedBuffer(on_host_shape_);
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalBuffer::DestructureTuple() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::DestructureTuple");
  absl::MutexLock lock(&mu_);
  if (!on_host_shape_.IsTuple()) {
    return InvalidArgument(
        "Attemped to destructure a PyLocalBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(on_host_shape_));
  }
  if (!device_buffer_) {
    return InvalidArgument("Attempted to destructure a deleted buffer.");
  }
  int num_children = ShapeUtil::TupleElementCount(on_host_shape_);
  std::vector<std::unique_ptr<PyLocalBuffer>> results;
  results.reserve(num_children);
  for (int64 i = 0; i < num_children; ++i) {
    results.push_back(absl::make_unique<PyLocalBuffer>(
        on_host_shape_.tuple_shapes(i), device_buffer_->children().at(i),
        client_));
  }
  return results;
}

Status PyLocalBuffer::BlockHostUntilReady() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::BlockHostUntilReady");
  std::shared_ptr<PySharedDeviceBuffer> device_buffer = DeviceBuffer();
  if (!device_buffer) {
    return InvalidArgument("BlockHostUntilReady() called on invalid buffer.");
  }

  client_->py_ref_manager().CollectGarbage();
  py::gil_scoped_release gil_release;

  // This code waits at least until the buffer is ready, but it may wait longer
  // if there are other device to host transfers scheduled. If this proves to
  // be an issue, we could either use a separate stream for this purpose, or
  // poll for the buffer definition events.
  se::Stream* stream =
      client_->device(device_buffer->device_ordinal()).device_to_host_stream();
  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  return stream->BlockHostUntilDone();
}

PyLocalExecutable::PyLocalExecutable(
    std::shared_ptr<LocalExecutable> executable,
    DeviceAssignment device_assignment, std::shared_ptr<PyLocalClient> client)
    : client_(std::move(client)),
      executable_(std::move(executable)),
      device_assignment_(std::move(device_assignment)) {}

std::vector<int> PyLocalExecutable::DeviceOrdinals() const {
  int num_replicas = device_assignment_.replica_count();
  std::vector<int> device_ordinals;
  device_ordinals.reserve(num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    device_ordinals.push_back(device_assignment_(i, 0));
  }
  return device_ordinals;
}

StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalExecutable::ExecuteHelper(
    absl::Span<PyLocalBuffer* const> argument_handles, int replica,
    const RunId& run_id) {
  const int device_ordinal = device_assignment_(replica, 0);
  tensorflow::profiler::TraceMe traceme("LocalExecutable::Execute");
  VLOG(3) << "Replica " << replica
          << " mapped to device ordinal for execution: " << device_ordinal;

  absl::flat_hash_set<BufferDefinitionEvent*> events;
  std::vector<std::shared_ptr<PySharedDeviceBuffer>> device_buffers;
  std::vector<ShapedBuffer> argument_buffers;
  std::vector<const ShapedBuffer*> argument_buffer_ptrs;
  device_buffers.reserve(argument_handles.size() + 1);
  argument_buffers.reserve(argument_handles.size());
  argument_buffer_ptrs.reserve(argument_handles.size());
  for (int i = 0; i < argument_handles.size(); ++i) {
    PyLocalBuffer* handle = argument_handles[i];
    std::shared_ptr<PySharedDeviceBuffer> device_buffer =
        handle->DeviceBuffer();
    if (!device_buffer) {
      return InvalidArgument(
          "Deleted buffer passed to Execute() as argument "
          "%d to replica %d",
          i, replica);
    }
    if (device_buffer->device_ordinal() != device_ordinal) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %d, but replica is assigned to device %d.",
          i, replica, device_buffer->device_ordinal(), device_ordinal);
    }
    TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, handle->AsShapedBuffer());
    argument_buffers.push_back(std::move(shaped_buffer));
    argument_buffer_ptrs.push_back(&argument_buffers.back());
    GetDeviceBufferDefinitionEvents(*device_buffer, &events);
    device_buffers.push_back(std::move(device_buffer));
    VLOG(4) << "Argument " << i
            << " buffer: " << argument_buffers.back().ToString();
  }

  const Device& device = client_->device(device_ordinal);
  // The choice of where we wait in "synchronous" mode is arbitrary; the reason
  // for the wait is pacing to avoid problems such as memory fragmentation, not
  // for correctness.
  if (!device.asynchronous()) {
    TF_RETURN_IF_ERROR(device.compute_stream()->BlockHostUntilDone());
  }

  for (BufferDefinitionEvent* event : events) {
    event->WaitForEventOnStream(device.compute_stream());
  }

  ExecutableRunOptions options;
  options.set_stream(device.compute_stream());
  options.set_host_to_device_stream(device.host_to_device_stream());
  options.set_allocator(client_->allocator());
  options.set_intra_op_thread_pool(
      client_->client()->backend().eigen_intra_op_thread_pool_device());
  options.set_device_assignment(&device_assignment_);
  options.set_run_id(run_id);

  StatusOr<ScopedShapedBuffer> result_buffer =
      executable_->RunAsync(argument_buffer_ptrs, options);

  VLOG(1) << "Replica " << replica << " completed; ok=" << result_buffer.ok();
  if (!result_buffer.ok()) {
    LOG(ERROR) << "Execution of replica " << replica
               << " failed: " << result_buffer.status();
    return result_buffer.status();
  }

  std::shared_ptr<BufferDefinitionEvent> definition_event;
  if (device.use_multiple_streams()) {
    TF_ASSIGN_OR_RETURN(
        definition_event,
        BufferDefinitionEvent::Create(device.compute_stream()->parent()));
    definition_event->RecordOnStream(device.compute_stream());
  }
  Shape on_host_shape = result_buffer.ValueOrDie().on_host_shape();
  std::shared_ptr<PySharedDeviceBuffer> out_buffer =
      PySharedDeviceBuffer::FromScopedShapedBuffer(
          std::move(result_buffer.ValueOrDie()), definition_event);

  if (device.synchronous_deallocation()) {
    device_buffers.push_back(out_buffer);
    device.ThenReleaseOnWorkerThread(device.compute_stream(),
                                     std::move(device_buffers));
  }
  device.ThenReleaseOnWorkerThread(device.compute_stream(), executable_);
  return absl::make_unique<PyLocalBuffer>(on_host_shape, std::move(out_buffer),
                                          client_);
}

StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalExecutable::Execute(
    absl::Span<PyLocalBuffer* const> argument_handles) {
  if (num_replicas() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d replicas using Execute()",
        num_replicas());
  }
  return ExecuteHelper(argument_handles, /*replica=*/0, RunId());
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalExecutable::ExecutePerReplica(
    absl::Span<const std::vector<PyLocalBuffer*>> argument_handles) {
  tensorflow::profiler::TraceMe traceme("LocalExecutable::ExecutePerReplica");
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

  VLOG(1) << "Executing replicated computation; num_replicas="
          << num_replicas();
  std::vector<StatusOr<std::unique_ptr<PyLocalBuffer>>> results(num_replicas());
  if (num_replicas() == 1) {
    // Fast-path if there is only one replica — run the computation on the
    // current thread.
    results[0] = ExecuteHelper(argument_handles[0], /*replica=*/0, RunId());
  } else {
    RunId run_id;
    absl::Mutex mu;
    int running GUARDED_BY(mu) = num_replicas();
    int failed GUARDED_BY(mu) = 0;
    Status first_failure_status GUARDED_BY(mu);

    for (int replica = 0; replica < num_replicas(); ++replica) {
      const int device_ordinal = device_assignment_(replica, 0);
      const Device& device = client_->device(device_ordinal);
      device.worker_thread()->Schedule([&, replica] {
        results[replica] =
            ExecuteHelper(argument_handles[replica], replica, run_id);

        absl::MutexLock lock(&mu);
        --running;
        if (!results[replica].ok()) {
          if (failed == 0) {
            first_failure_status = results[replica].status();
          }
          ++failed;
        }
      });
    }

    auto done_running_or_failed = [&]() {
      mu.AssertHeld();
      return running == 0 || failed > 0;
    };
    absl::MutexLock lock(&mu);
    mu.Await(absl::Condition(&done_running_or_failed));
    if (failed > 0) {
      auto done_running = [&]() {
        mu.AssertHeld();
        return running == 0;
      };
      // If execution does not terminate within a reasonable amount of time, we
      // may be stuck at a cross-replica barrier on-device. Terminate the
      // process since that's the only way we can escape this situation at the
      // moment (b/130629719).
      if (!mu.AwaitWithTimeout(absl::Condition(&done_running),
                               absl::Seconds(10))) {
        LOG(FATAL)
            << "Replicated computation launch failed, but not all replicas "
               "terminated. Aborting process to work around deadlock. Failure "
               "message (there may have been multiple failures, see the "
               "error log for all failures): \n\n"
            << first_failure_status.error_message();
      }
    }
  }
  VLOG(1) << "Replicated execution complete.";

  std::vector<std::unique_ptr<PyLocalBuffer>> wrapped_results(num_replicas());
  for (int replica = 0; replica < num_replicas(); ++replica) {
    auto& statusor = results[replica];
    if (!statusor.ok()) {
      return AppendStatus(
          statusor.status(),
          absl::StrFormat(
              "while running replica %d of a replicated computation (other "
              "replicas may have failed as well).",
              replica));
    }
    wrapped_results[replica] = std::move(statusor.ValueOrDie());
  }
  return wrapped_results;
}

/*static*/ StatusOr<std::unique_ptr<PyLocalExecutable>>
PyLocalExecutable::Compile(const XlaComputation& computation,
                           absl::optional<std::vector<Shape>> argument_layouts,
                           const ExecutableBuildOptions* build_options,
                           std::shared_ptr<PyLocalClient> client,
                           absl::optional<DeviceAssignment> device_assignment) {
  tensorflow::profiler::TraceMe traceme("LocalExecutable::Compile");

  ExecutableBuildOptions options;
  if (build_options != nullptr) {
    options = *build_options;
  }

  if (!options.device_allocator()) {
    options.set_device_allocator(client->allocator());
  }

  if (device_assignment) {
    if (device_assignment->replica_count() != options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).",
          device_assignment->replica_count(), options.num_replicas());
    } else if (device_assignment->computation_count() != 1) {
      return Unimplemented(
          "Only 1 computation per replica supported, %d requested.",
          device_assignment->computation_count());
    }
  } else {
    TF_ASSIGN_OR_RETURN(
        device_assignment,
        client->client()->backend().computation_placer()->AssignDevices(
            options.num_replicas(), /*computation_count=*/1));
  }

  if (!argument_layouts) {
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        computation.GetProgramShape());
    argument_layouts = program_shape.parameters();
    for (Shape& shape : *argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  }
  std::vector<const Shape*> argument_layout_pointers;
  argument_layout_pointers.reserve(argument_layouts->size());

  // Assign a default layout to any array subshapes that are missing layouts.
  auto assign_layouts = [client](Shape* shape) {
    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        shape, [&](Shape* subshape, const ShapeIndex&) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(*subshape,
                                client->client()
                                    ->backend()
                                    .transfer_manager()
                                    ->ChooseCompactLayoutForShape(*subshape));
          }
          return Status::OK();
        });
  };

  for (Shape& layout : *argument_layouts) {
    argument_layout_pointers.push_back(&layout);
    TF_RETURN_IF_ERROR(assign_layouts(&layout));
  }

  Shape result_layout;
  if (options.result_layout()) {
    result_layout = *options.result_layout();
  } else {
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        computation.GetProgramShape());
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
  }
  TF_RETURN_IF_ERROR(assign_layouts(&result_layout));
  options.set_result_layout(result_layout);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<LocalExecutable> local_executable,
                      client->client()->Compile(
                          computation, argument_layout_pointers, options));

  return absl::make_unique<PyLocalExecutable>(
      std::shared_ptr<LocalExecutable>(std::move(local_executable)),
      std::move(*device_assignment), std::move(client));
}

}  // namespace xla
