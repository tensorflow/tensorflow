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

#include "tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {

TpuDevice::TpuDevice(int id, int process_index,
                     const std::array<int, 3>& coords, int core_on_chip)
    : id_(id),
      process_index_(process_index),
      coords_(coords),
      core_on_chip_(core_on_chip) {}

std::string TpuDevice::DebugString() const {
  return absl::StrFormat("TPU_%i(host=%i,(%i,%i,%i,%i))", id(), process_index(),
                         coords_[0], coords_[1], coords_[2], core_on_chip_);
}

std::string TpuDevice::ToString() const {
  return absl::StrFormat(
      "TpuDevice(id=%i, process_index=%i, coords=(%s), core_on_chip=%i)", id(),
      process_index(), absl::StrJoin(coords(), ","), core_on_chip());
}

xla::StatusOr<std::vector<std::shared_ptr<xla::PjRtDevice>>>
TpuDevice::GetTpuDevices(const tpu_driver::SystemInfo& system_info) {
  std::vector<std::shared_ptr<PjRtDevice>> devices;
  for (const auto& chip : system_info.tpu_chip()) {
    auto& coord = chip.chip_coord();
    std::array<int, 3> coords_array = {coord.x(), coord.y(), coord.z()};
    int process_index = chip.host_id();
    for (const auto& core : chip.core()) {
      auto device = std::make_shared<TpuDevice>(
          core.id(), process_index, coords_array, core.core_on_chip_index());
      devices.push_back(device);
    }
  }

  return devices;
}

StatusOr<std::shared_ptr<PyTpuClient>> PyTpuClient::Get(
    const std::string& worker) {
  tpu_driver::TpuDriverConfig driver_config;
  driver_config.set_worker(worker);
  auto client_status = tpu_driver::TpuDriverRegistry::Open(driver_config);
  if (!client_status.ok()) {
    return client_status.status();
  }

  auto client = client_status.ConsumeValueOrDie();

  tpu_driver::SystemInfo system_info;
  client->QuerySystemInfo(&system_info);

  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<PjRtDevice>> devices,
                      TpuDevice::GetTpuDevices(system_info));

  return std::make_shared<PyTpuClient>(TpuPlatform(), std::move(client),
                                       std::move(devices),
                                       system_info.host_id());
}

PyTpuClient::PyTpuClient(std::string platform_name,
                         std::unique_ptr<tpu_driver::TpuDriver> driver,
                         std::vector<std::shared_ptr<PjRtDevice>> devices,
                         int process_index)
    : platform_name_(std::move(platform_name)),
      platform_version_("tpu_driver (deprecated)"),
      driver_(std::move(driver)),
      devices_(std::move(devices)),
      process_index_(process_index) {
  for (const std::shared_ptr<PjRtDevice>& device : devices_) {
    tensorflow::down_cast<TpuDevice*>(device.get())->set_tpu_client(this);
    CHECK(id_to_device_.insert({device->id(), device}).second)
        << "Duplicate device id: " << device->id();

    if (device->process_index() == process_index_) {
      LOG(INFO) << "Detected local device, host id: " << process_index_
                << ". device id: " << device->id();
      local_devices_.push_back(device);
    } else {
      VLOG(2) << "Other devices, device id: " << device->id();
    }
  }
  CHECK_GE(local_devices_.size(), 1);
  LOG(INFO) << "Creating " << local_devices_.size() << " TPU device(s).";

  for (int idx = 0; idx < local_devices_.size(); ++idx) {
    CHECK(local_devices_[idx] != nullptr) << idx;
  }

  // TODO(frankchn): Check if thread pool size needs to be adjusted (perhaps
  // something like min(cores, devices_.size()) might be appropriate depending
  // on the number of devices.
  pool_ = std::make_unique<tensorflow::thread::ThreadPool>(
      tensorflow::Env::Default(), "PyTpuClient", devices_.size());
}

Status PyTpuClient::TransferToInfeed(const LiteralSlice& literal,
                                     int device_id) {
  return Unimplemented("Infeed not implemented.");
}

StatusOr<Literal> PyTpuClient::TransferFromOutfeed(const Shape& shape,
                                                   int device_id) {
  return Unimplemented("Outfeed not implemented.");
}

StatusOr<DeviceAssignment> PyTpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_partitions > 1) {
    return InvalidArgument("Num partitions greater than 1, is not supported.");
  }
  if (num_replicas * num_partitions <= local_device_count()) {
    DeviceAssignment assignment(num_replicas, num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        assignment(replica, partition) = local_devices_[replica]->id();
      }
    }
    return assignment;
  }

  // Fallback to default global device assignment if we can't run locally.
  xla::ComputationPlacer placer;
  return placer.AssignDevices(num_replicas, num_partitions);
}

Status PyTpuClient::CheckDeviceId(int device_id,
                                  absl::string_view caller_name) {
  if (device_id < 0 || device_id >= device_count()) {
    return InvalidArgument("%s got bad device_id: %d (num_devices=%d).",
                           caller_name, device_id, device_count());
  }
  return OkStatus();
}

static Status CheckDataType(xla::PrimitiveType dtype) {
  if (dtype == xla::PrimitiveType::F64 || dtype == xla::PrimitiveType::S64 ||
      dtype == xla::PrimitiveType::U64) {
    return InvalidArgument(
        "64-bit data types are not yet supported on the TPU driver API. "
        "Convert inputs to float32/int32_t before using.");
  }
  return OkStatus();
}

/* static */
StatusOr<std::unique_ptr<PyTpuBuffer>> PyTpuBuffer::FromLiterals(
    std::vector<BorrowingLiteral> leaves, const Shape& tuple_shape,
    std::shared_ptr<void> leaves_references,
    std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device) {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::FromLiterals");
  VLOG(1) << "PyTpuBuffer::FromLiterals: shape: " << tuple_shape.DebugString()
          << " device: " << device->DebugString();
  TF_RETURN_IF_ERROR(
      client->CheckDeviceId(device->id(), "PyTpuBuffer::FromLiterals"));
  tpu_driver::TpuDriver* driver = client->driver();

  if (!tuple_shape.IsTuple()) {
    TF_RET_CHECK(leaves.size() == 1);
    return CreateBuffer(
        tuple_shape,
        [driver, &leaves, &tuple_shape,
         leaves_references](tpu_driver::BufferHandle* handle) {
          auto event =
              driver->TransferToDevice(leaves[0].untyped_data(), handle, {});
          event->AddCallback([leaves_references](Status) {});
          return event;
        },
        std::move(client), std::move(device));
  }

  std::vector<std::unique_ptr<PyTpuBuffer>> child_buffers;
  child_buffers.reserve(leaves.size());
  std::vector<PyTpuBuffer*> child_buffer_ptrs;
  child_buffer_ptrs.reserve(leaves.size());

  auto it_leaf = leaves.begin();
  for (const ShapeUtil::IndexedShape& indexed_shape :
       ShapeUtil::GetLeafShapes(tuple_shape)) {
    TF_RET_CHECK(it_leaf != leaves.end());
    auto& leaf = *it_leaf;
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PyTpuBuffer> child_buffer,
        CreateBuffer(
            indexed_shape.shape,
            [driver, &leaf, &indexed_shape](tpu_driver::BufferHandle* handle) {
              return driver->TransferToDevice(leaf.untyped_data(), handle, {});
            },
            client, device));
    child_buffer_ptrs.push_back(child_buffer.get());
    child_buffers.push_back(std::move(child_buffer));
    ++it_leaf;
  }
  TF_RET_CHECK(it_leaf == leaves.end());

  // `MakeTuple` will extract and make the tuple buffer hold onto the
  // `device_buffer_` contained in each `child_buffer`, so it's safe for
  // `child_buffers` to get destroyed before this call returns.
  return MakeTuple(std::move(child_buffer_ptrs), std::move(client),
                   std::move(device));
}

/* static */
StatusOr<std::unique_ptr<PyTpuBuffer>> PyTpuBuffer::MakeTuple(
    absl::Span<PyTpuBuffer* const> buffers, std::shared_ptr<PyTpuClient> client,
    std::shared_ptr<PjRtDevice> device) {
  std::vector<Shape> child_shapes;
  std::vector<std::shared_ptr<TpuSharedBuffer>> child_device_buffers;
  std::vector<tpu_driver::BufferHandle*> child_handle_ptrs;
  std::vector<std::shared_ptr<tpu_driver::Event>> child_events;

  for (const auto& child_buffer : buffers) {
    child_shapes.push_back(child_buffer->on_host_shape());
    std::shared_ptr<TpuSharedBuffer> child_device_buffer =
        child_buffer->DeviceBuffer();
    // Merge all definition events from all children, so that anyone using this
    // tuple must wait for all its children to finish receiving transfers. This
    // works recursively up a nested tuple tree as well.
    for (std::shared_ptr<tpu_driver::Event> child_event :
         child_device_buffer->wait_for_use) {
      child_events.push_back(std::move(child_event));
    }
    child_handle_ptrs.push_back(child_device_buffer->handle.get());
    child_device_buffers.push_back(std::move(child_device_buffer));
  }

  Shape tuple_shape = ShapeUtil::MakeTupleShape(child_shapes);
  std::unique_ptr<tpu_driver::BufferHandle> tuple_handle =
      client->driver()->AllocateTuple(
          device->id(), tpu_driver::MemoryRegion::HBM, child_handle_ptrs, {});
  auto tuple_device_buffer = std::make_shared<TpuSharedBuffer>(
      client->driver(), std::move(tuple_handle), std::move(child_events),
      std::move(device));
  return absl::make_unique<PyTpuBuffer>(
      tuple_shape, std::move(tuple_device_buffer),
      std::move(child_device_buffers), std::move(client));
}

PyTpuBuffer::PyTpuBuffer(
    Shape on_host_shape, std::shared_ptr<TpuSharedBuffer> device_buffer,
    std::vector<std::shared_ptr<TpuSharedBuffer>> child_buffers,
    std::shared_ptr<PyTpuClient> client)
    : client_(std::move(client)),
      on_host_shape_(std::move(on_host_shape)),
      device_(device_buffer->device),
      device_buffer_(std::move(device_buffer)),
      child_buffers_(std::move(child_buffers)) {}

void PyTpuBuffer::Delete() {
  absl::MutexLock lock(&mu_);
  device_buffer_ = nullptr;
  child_buffers_.clear();
  host_value_ = nullptr;
}

Status PyTpuBuffer::CopyToHostAsync() {
  std::vector<std::shared_ptr<tpu_driver::Event>> transfer_events;
  std::shared_ptr<HostValue> host_value = std::make_shared<HostValue>();

  {
    absl::MutexLock lock(&mu_);
    if (!device_buffer_) {
      return InvalidArgument("CopyToHostAsync() called on invalid buffer.");
    }

    if (host_value_) {
      // The host value has already been requested or is available.
      return OkStatus();
    }

    host_value->value = std::make_shared<Literal>(on_host_shape_);
    host_value->pending_ops = std::max(1ul, child_buffers_.size());
    host_value_ = host_value;

    std::vector<tpu_driver::Event*> events;
    for (const auto& e : device_buffer_->wait_for_use) {
      events.push_back(e.get());
    }

    VLOG(1) << "CopyToHostAsync:: host shape: "
            << host_value->value->shape().DebugString();

    if (!on_host_shape_.IsTuple()) {
      CHECK(child_buffers_.empty());
      transfer_events.push_back(client_->driver()->TransferFromDevice(
          device_buffer_->handle.get(), host_value->value->untyped_data(),
          events));
    } else {
      for (int i = 0; i < child_buffers_.size(); ++i) {
        auto& c = child_buffers_[i];
        transfer_events.push_back(client_->driver()->TransferFromDevice(
            c->handle.get(),
            host_value->value->untyped_data(xla::ShapeIndex({i})), events));
      }
    }
  }

  for (auto& t : transfer_events) {
    t->AddCallback([host_value](const xla::Status& status) {
      VLOG(1) << "Device to host transfer finished.";
      if (!status.ok()) {
        host_value->status =
            Status(static_cast<tensorflow::error::Code>(status.code()),
                   status.error_message());
      }

      absl::MutexLock m(&host_value->mutex);
      --host_value->pending_ops;
      if (host_value->pending_ops == 0) {
        VLOG(1) << "Host value done: " << host_value->status;
        host_value->ready.Notify();
      }
    });
  }
  return OkStatus();
}

StatusOr<std::shared_ptr<Literal>> PyTpuBuffer::ToLiteral() {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::ToLiteral");
  TF_RETURN_IF_ERROR(CopyToHostAsync());

  mu_.Lock();
  std::shared_ptr<HostValue> host_value = host_value_;
  mu_.Unlock();

  VLOG(1) << "Waiting for device to host transfer " << host_value.get();
  host_value->ready.WaitForNotification();
  VLOG(1) << "Host copy finished, status:" << host_value->status;
  TF_RETURN_IF_ERROR(host_value->status);

  return host_value->value;
}

std::shared_ptr<TpuSharedBuffer> PyTpuBuffer::DeviceBuffer() const {
  absl::MutexLock lock(&mu_);
  return device_buffer_;
}

StatusOr<std::vector<std::unique_ptr<PyTpuBuffer>>>
PyTpuBuffer::DestructureTuple() {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::DestructureTuple");
  if (!on_host_shape_.IsTuple()) {
    return InvalidArgument(
        "Attempted to destructure a PyTpuBuffer that did not have a tuple "
        "shape; shape: %s.",
        ShapeUtil::HumanString(on_host_shape_));
  }
  if (DeviceBuffer() == nullptr) {
    return InvalidArgument("Attempted to destructure a deleted buffer.");
  }

  absl::MutexLock lock(&mu_);
  int num_children = ShapeUtil::TupleElementCount(on_host_shape_);
  std::vector<std::unique_ptr<PyTpuBuffer>> results;
  results.reserve(num_children);
  for (int i = 0; i < num_children; ++i) {
    results.push_back(absl::make_unique<PyTpuBuffer>(
        on_host_shape_.tuple_shapes(i), child_buffers_.at(i),
        std::vector<std::shared_ptr<TpuSharedBuffer>>(), client_));
  }
  return results;
}

StatusOr<std::unique_ptr<PyTpuBuffer>> PyTpuBuffer::CopyToDevice(
    std::shared_ptr<PjRtDevice> dst_device) {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::CopyToDevice");
  if (on_host_shape_.IsTuple()) {
    return Unimplemented("CopyToDevice for tuples is not supported.");
  }

  std::shared_ptr<TpuSharedBuffer> src_device_buffer = DeviceBuffer();
  if (dst_device->id() == device_->id()) {
    return absl::make_unique<PyTpuBuffer>(
        on_host_shape_, src_device_buffer,
        std::vector<std::shared_ptr<TpuSharedBuffer>>(), client_);
  }

  tpu_driver::TpuDriver* driver = client_->driver();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PyTpuBuffer> dst_buffer,
      CreateBuffer(
          on_host_shape_,
          [driver, src_device_buffer](tpu_driver::BufferHandle* dst_handle) {
            std::vector<tpu_driver::Event*> src_wait_for_use;
            for (auto& event : src_device_buffer->wait_for_use) {
              src_wait_for_use.push_back(event.get());
            }
            return driver->TransferFromDeviceToDevice(
                src_device_buffer->handle.get(), dst_handle, src_wait_for_use);
          },
          client_, std::move(dst_device)));
  // TODO(jiawenhao): This may be too pessimistic: it prevents future readers
  // from reading `src_device_buffer` until the device-to-device copy is done.
  // Should this go into a new `TpuSharedBuffer::wait_for_dealloc` field?
  auto& wait_for_use = dst_buffer->DeviceBuffer()->wait_for_use;
  src_device_buffer->wait_for_use.insert(src_device_buffer->wait_for_use.end(),
                                         wait_for_use.begin(),
                                         wait_for_use.end());
  return dst_buffer;
}

Status PyTpuBuffer::BlockHostUntilReady() {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::BlockHostUntilReady");
  std::shared_ptr<TpuSharedBuffer> device_buffer = DeviceBuffer();
  if (!device_buffer) {
    return InvalidArgument(
        "BlockHostUntilReady() called on deleted or donated buffer");
  }
  return device_buffer->handle->OnReady()->Await();
}

/* static */
StatusOr<std::unique_ptr<PyTpuBuffer>> PyTpuBuffer::AllocateBuffer(
    const Shape& shape, std::shared_ptr<PyTpuClient> client,
    std::shared_ptr<PjRtDevice> device) {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::AllocateBuffer");
  VLOG(1) << "PyTpuBuffer::AllocateBuffer: shape: " << shape.DebugString()
          << " device: " << device->DebugString();

  if (!shape.IsTuple()) {
    return CreateBuffer(shape, absl::nullopt, std::move(client),
                        std::move(device));
  }

  std::vector<std::unique_ptr<PyTpuBuffer>> child_buffers;
  child_buffers.reserve(shape.tuple_shapes().size());
  std::vector<PyTpuBuffer*> child_buffer_ptrs;
  child_buffer_ptrs.reserve(shape.tuple_shapes().size());

  for (const auto& child_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PyTpuBuffer> child_buffer,
                        AllocateBuffer(child_shape, client, device));
    child_buffer_ptrs.push_back(child_buffer.get());
    child_buffers.push_back(std::move(child_buffer));
  }

  // `MakeTuple` will extract and make the tuple buffer hold onto the
  // `device_buffer_` contained in each `child_buffer`, so it's safe for
  // `child_buffers` to get destroyed before this call returns.
  return PyTpuBuffer::MakeTuple(child_buffer_ptrs, std::move(client),
                                std::move(device));
}

/*static*/
StatusOr<std::unique_ptr<PyTpuBuffer>> PyTpuBuffer::CreateBuffer(
    const Shape& non_tuple_shape, absl::optional<BufferInitializer> initializer,
    std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device) {
  tensorflow::profiler::TraceMe traceme("PyTpuBuffer::CreateBuffer");
  VLOG(1) << "PyTpuBuffer::CreateBuffer: shape: "
          << non_tuple_shape.DebugString()
          << " device: " << device->DebugString();
  TF_RET_CHECK(!non_tuple_shape.IsTuple());
  TF_RETURN_IF_ERROR(CheckDataType(non_tuple_shape.element_type()));

  std::unique_ptr<tpu_driver::BufferHandle> handle =
      client->driver()->Allocate(device->id(), tpu_driver::MemoryRegion::HBM,
                                 non_tuple_shape.ToProto(), {});

  // If this buffer needs to be initialized, anyone using this buffer must wait
  // for the initialization event in `wait_for_use` to finish first.
  std::vector<std::shared_ptr<tpu_driver::Event>> wait_for_use;
  if (initializer.has_value()) {
    std::shared_ptr<tpu_driver::Event> init = initializer.value()(handle.get());
    wait_for_use.push_back(std::move(init));
  }
  auto device_buffer = std::make_shared<TpuSharedBuffer>(
      client->driver(), std::move(handle), std::move(wait_for_use),
      std::move(device));

  return absl::make_unique<PyTpuBuffer>(
      non_tuple_shape, std::move(device_buffer),
      std::vector<std::shared_ptr<TpuSharedBuffer>>(), client);
}

static std::shared_ptr<PjRtDevice> LookupDevice(const PyTpuClient& client,
                                                int device_id) {
  auto it = client.id_to_device().find(device_id);
  CHECK(it != client.id_to_device().end())
      << "Unknown device id: " << device_id;
  return it->second;
}

PyTpuExecutable::PyTpuExecutable(
    std::unique_ptr<tpu_driver::CompiledProgramHandle> compiled_program,
    DeviceAssignment device_assignment, std::shared_ptr<PyTpuClient> client,
    xla::Shape result_shape, bool tuple_arguments)
    : client_(std::move(client)),
      device_assignment_(std::move(device_assignment)),
      tuple_arguments_(tuple_arguments),
      result_shape_(std::move(result_shape)) {
  VLOG(1) << "DeviceAssignment. " << device_assignment_.ToString();
  const int num_replicas = device_assignment_.replica_count();
  const int num_partitions = device_assignment_.computation_count();
  CHECK_EQ(num_partitions, 1) << "partition count > 1 is not supported.";
  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int partition = 0; partition < num_partitions; ++partition) {
      int device_id = device_assignment_(replica, partition);
      std::shared_ptr<PjRtDevice> device = LookupDevice(*client_, device_id);
      if (device->process_index() != client_->process_index()) {
        VLOG(3) << "Non-local device: " << device_id;
        continue;
      }
      // TODO(b/147895917): support replica + partition natively.
      CHECK(executables_.find(replica) == executables_.end())
          << "Inserting duplicate replica:" << replica;
      executables_[replica] =
          client_->driver()->LoadProgram(device_id, compiled_program.get(), {});
      local_logical_device_ids_.emplace_back(replica, partition);
      local_devices_.push_back(device);
    }
  }
  CHECK_GE(local_devices_.size(), 1);
  CHECK_LE(executables_.size(), client_->device_count());
  CHECK_LE(local_devices_.size(), client_->local_device_count())
      << "Inconsistent local device count.";
}

PyTpuExecutable::ExecuteResult PyTpuExecutable::ExecuteHelper(
    absl::Span<const std::vector<PyTpuBuffer*>> maybe_tupled_args,
    absl::Span<PyTpuBuffer* const> this_core_arguments, int replica,
    int partition, const RunId& run_id) {
  const int device_id = device_assignment_(replica, partition);
  std::shared_ptr<PjRtDevice> device = LookupDevice(*client_, device_id);
  CHECK_EQ(device->process_index(), client_->process_index());
  tensorflow::profiler::TraceMe traceme("PyTpuExecutable::Execute");
  VLOG(3) << "Replica " << replica << ", partition " << partition
          << " mapped to device id for execution: " << device_id;

  std::unique_ptr<::xla::PyTpuBuffer> output_buffer =
      ::xla::PyTpuBuffer::AllocateBuffer(result_shape_, client_,
                                         std::move(device))
          .ValueOrDie();
  VLOG(1) << "Created output buffer: " << result_shape_.DebugString();

  std::vector<tpu_driver::BufferHandle*> inputs;
  std::vector<tpu_driver::Event*> ready_to_execute;

  std::shared_ptr<tpu_driver::Event> output_buffer_ready =
      output_buffer->DeviceBuffer()->handle->OnReady();

  ready_to_execute.push_back(output_buffer_ready.get());

  for (auto* input_handle : this_core_arguments) {
    inputs.push_back(input_handle->DeviceBuffer()->handle.get());
  }

  for (const auto& core_args : maybe_tupled_args) {
    for (const auto* handle : core_args) {
      for (const auto& pending_event : handle->DeviceBuffer()->wait_for_use) {
        ready_to_execute.push_back(pending_event.get());
      }
    }
  }

  xla::DeviceAssignmentProto device_assignment;
  CHECK(device_assignment_.Serialize(&device_assignment).ok());
  std::shared_ptr<tpu_driver::Event> on_execute_finished =
      client_->driver()->ExecuteProgram(
          executables_.find(replica)->second.get(), inputs,
          {output_buffer->DeviceBuffer()->handle.get()}, device_assignment,
          {ready_to_execute});

  return {std::move(output_buffer), std::move(on_execute_finished)};
}

// Delay before warning about a slow execute call.
static const absl::Duration kWarnExecutionDelay = absl::Seconds(10);

// Delay before terminating a stalled execute call.
static const absl::Duration kMaxExecutionDelay = absl::Minutes(60);

Status WaitForExecuteEvent(tpu_driver::Event* event) {
  absl::optional<Status> opt_status;
  auto start_time = absl::Now();

  while (!opt_status.has_value() &&
         absl::Now() - start_time < kMaxExecutionDelay) {
    opt_status = event->AwaitWithTimeout(kWarnExecutionDelay);
    if (!opt_status.has_value()) {
      LOG(WARNING)
          << "TPU Execute is taking a long time. This might be due to a "
             "deadlock between multiple TPU cores or a very slow program.";
    }
  }

  if (!opt_status.has_value()) {
    return tensorflow::errors::DeadlineExceeded(
        absl::StrFormat("TPU program took more than %d seconds to complete.",
                        absl::ToInt64Seconds(kMaxExecutionDelay)));
  }

  return opt_status.value();
}

StatusOr<std::vector<std::unique_ptr<PyTpuBuffer>>> PyTpuExecutable::Execute(
    absl::Span<PyTpuBuffer* const> argument_handles) {
  if (num_replicas() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d replicas using Execute().",
        num_replicas());
  }
  if (num_partitions() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d partitions using Execute().",
        num_partitions());
  }

  std::vector<PyTpuBuffer*> maybe_tupled_args;
  std::unique_ptr<PyTpuBuffer> tupled_arguments;
  if (tuple_arguments_) {
    TF_ASSIGN_OR_RETURN(tupled_arguments,
                        PyTpuBuffer::MakeTuple(argument_handles, client_,
                                               local_devices_.front()));
    maybe_tupled_args = {tupled_arguments.get()};
  } else {
    maybe_tupled_args = std::vector<PyTpuBuffer*>(argument_handles.begin(),
                                                  argument_handles.end());
  }
  ExecuteResult result =
      ExecuteHelper(absl::MakeSpan(&maybe_tupled_args, 1), maybe_tupled_args,
                    /*replica=*/0, /*partition=*/0, RunId());

  Status status = WaitForExecuteEvent(result.on_execute_finished.get());

  if (!status.ok()) {
    LOG(ERROR) << "Failed to execute program: " << status;
    return status;
  }

  if (result.buffer->on_host_shape().IsTuple()) {
    return result.buffer->DestructureTuple();
  } else {
    std::vector<std::unique_ptr<PyTpuBuffer>> outputs;
    outputs.push_back(std::move(result.buffer));
    return outputs;
  }
}

StatusOr<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>>
PyTpuExecutable::ExecuteOnLocalDevices(
    absl::Span<const std::vector<PyTpuBuffer*>> argument_handles) {
  tensorflow::profiler::TraceMe traceme(
      "PyTpuExecutable::ExecuteOnLocalDevices");

  const int num_local_devices = local_devices_.size();

  if (argument_handles.size() != num_local_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d).",
        argument_handles.size(), num_local_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions()
          << " num_local_devices=" << num_local_devices;

  std::vector<std::unique_ptr<PyTpuBuffer>> tupled_arguments;
  std::vector<std::vector<PyTpuBuffer*>> tupled_argument_pointers;
  if (tuple_arguments_) {
    tupled_arguments.resize(argument_handles.size());
    tupled_argument_pointers.resize(argument_handles.size());
    for (int i = 0; i < num_local_devices; ++i) {
      TF_ASSIGN_OR_RETURN(tupled_arguments[i],
                          PyTpuBuffer::MakeTuple(argument_handles[i], client_,
                                                 local_devices_.at(i)));
      tupled_argument_pointers[i] = {tupled_arguments[i].get()};
    }
    argument_handles = tupled_argument_pointers;
  }

  absl::Mutex results_lock;
  std::vector<ExecuteResult> results(num_local_devices);

  auto* thread_pool = client_->GetThreadPool();

  int failed = 0;
  Status first_failure_status;

  xla::Semaphore execute_semaphore(0);
  for (int i = 0; i < num_local_devices; ++i) {
    // We are scheduling Execute on a thread pool as ExecuteHelper can take a
    // long time and we want all cores to be scheduled in parallel.
    thread_pool->Schedule([this, i, argument_handles, &results, &results_lock,
                           &execute_semaphore]() {
      const int replica = local_logical_device_ids_[i].first;
      const int partition = local_logical_device_ids_[i].second;
      RunId run_id;
      auto result = ExecuteHelper(argument_handles, argument_handles[i],
                                  replica, partition, run_id);
      results[i] = std::move(result);
      execute_semaphore.Release(1);
    });
  }

  execute_semaphore.Acquire(num_local_devices);

  for (int i = 0; i < num_local_devices; ++i) {
    auto s = WaitForExecuteEvent(results[i].on_execute_finished.get());
    if (!s.ok()) {
      if (failed == 0) {
        first_failure_status = Status(
            static_cast<tensorflow::error::Code>(s.code()), s.error_message());
      }
      ++failed;
    }
  }
  if (failed > 0) {
    return first_failure_status;
  }
  VLOG(1) << "Replicated execution complete.";

  std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>> wrapped_results(
      num_local_devices);
  for (int i = 0; i < num_local_devices; ++i) {
    if (results[i].buffer->on_host_shape().IsTuple()) {
      TF_ASSIGN_OR_RETURN(wrapped_results[i],
                          results[i].buffer->DestructureTuple());
    } else {
      wrapped_results[i].push_back(std::move(results[i].buffer));
    }
  }
  return wrapped_results;
}

StatusOr<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>>
PyTpuExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const std::vector<PyTpuBuffer*>> args) {
  std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>> output_buffers;
  TF_RET_CHECK(!args.empty());
  int num_computations = args.front().size();
  for (const auto& arg : args) {
    if (arg.size() != num_computations) {
      return xla::InvalidArgument(
          "Expected args to execute_sharded_on_local_devices to have %d "
          "shards, got: [%s]",
          num_computations,
          absl::StrJoin(
              args, ", ",
              [](std::string* out, const std::vector<PyTpuBuffer*>& arg) {
                out->append(std::to_string(arg.size()));
              }));
    }
  }
  std::vector<std::vector<PyTpuBuffer*>> arg_buffers(num_computations);
  for (int computation = 0; computation < num_computations; ++computation) {
    arg_buffers[computation].resize(args.size());
    absl::c_transform(
        args, arg_buffers[computation].begin(),
        [&](const std::vector<PyTpuBuffer*>& arg) { return arg[computation]; });
  }
  TF_ASSIGN_OR_RETURN(output_buffers, ExecuteOnLocalDevices(arg_buffers));
  int num_output_buffers = output_buffers[0].size();
  std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>> outputs;
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      outputs[buffer_id].push_back(
          std::move(output_buffers[computation][buffer_id]));
    }
  }
  return outputs;
}

/*static*/ StatusOr<std::unique_ptr<PyTpuExecutable>> PyTpuExecutable::Compile(
    const XlaComputation& computation,
    absl::optional<std::vector<Shape>> argument_layouts,
    const ExecutableBuildOptions* build_options,
    std::shared_ptr<PyTpuClient> client, bool tuple_arguments) {
  tensorflow::profiler::TraceMe traceme("PyTpuExecutable::Compile");

  VLOG(1) << "Compile: "
          << computation.GetProgramShape().ValueOrDie().DebugString();

  // TODO(power) -- handle argument layouts
  // TODO(power) -- handle build options
  ExecutableBuildOptions options;
  if (build_options != nullptr) {
    options = *build_options;
  }
  absl::optional<xla::DeviceAssignment> device_assignment;

  // For POD use case, the device_assignment.num_replicas() may be greater than
  // the number of available local devices, where applicable the non-local
  // devices must be filtered out from participating local computation.
  if (options.has_device_assignment()) {
    if (options.device_assignment().replica_count() != options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).",
          options.device_assignment().replica_count(), options.num_replicas());
    } else if (options.device_assignment().computation_count() != 1) {
      return Unimplemented(
          "Only 1 computation per replica supported, %d requested.",
          options.device_assignment().computation_count());
    }
    device_assignment = options.device_assignment();
  } else {
    TF_ASSIGN_OR_RETURN(device_assignment,
                        client->GetDefaultDeviceAssignment(
                            options.num_replicas(), options.num_partitions()));
  }
  CHECK_GE(options.num_replicas(), 1);
  CHECK_EQ(options.num_replicas(), device_assignment->replica_count());
  CHECK(!argument_layouts.has_value());

  // TODO(henrytan): an area for optimization with less buffer copy.
  xla::HloProto hlo_proto;
  *hlo_proto.mutable_hlo_module() = computation.proto();

  // TODO(henrytan): in the future, we want to consider argument Layout
  // information e.g. for linearization.
  std::unique_ptr<tpu_driver::CompiledProgramHandle> compiled_program =
      client->driver()->CompileProgram(hlo_proto, options.num_replicas(), {});

  ::xla::Shape result_layout;
  if (options.result_layout()) {
    result_layout = *options.result_layout();
  } else {
    xla::ProgramShapeProto program_shape_proto;
    auto fetch_metadata_status =
        compiled_program->program_shape(&program_shape_proto);

    if (!fetch_metadata_status.ok()) {
      return Status(
          static_cast<tensorflow::error::Code>(fetch_metadata_status.code()),
          fetch_metadata_status.error_message());
    }
    result_layout = ::xla::Shape(program_shape_proto.result());
  }
  VLOG(1) << "Got result shape: " << result_layout.DebugString();

  return absl::make_unique<PyTpuExecutable>(
      std::move(compiled_program), std::move(*device_assignment),
      std::move(client), std::move(result_layout), tuple_arguments);
}

/*static*/ StatusOr<std::unique_ptr<PyTpuExecutable>>
PyTpuExecutable::CompileMlir(
    mlir::ModuleOp module, absl::optional<std::vector<Shape>> argument_layouts,
    const ExecutableBuildOptions* build_options,
    std::shared_ptr<PyTpuClient> client, bool tuple_arguments) {
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(module, xla_computation,
                                          /*use_tuple_args=*/tuple_arguments,
                                          /*return_tuple=*/false));
  return Compile(xla_computation, argument_layouts, build_options, client,
                 tuple_arguments);
}

}  // namespace xla
