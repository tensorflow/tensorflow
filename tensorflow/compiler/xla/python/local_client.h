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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/python/local_device_state.h"
#include "tensorflow/compiler/xla/python/shared_device_buffer.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

class Device {
 public:
  explicit Device(int id, std::unique_ptr<LocalDeviceState> local_device_state,
                  absl::string_view platform_name, int host_id = 0)
      : id_(id),
        local_device_state_(std::move(local_device_state)),
        host_id_(host_id),
        platform_name_(platform_name) {}
  virtual ~Device() {}

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  int id() const { return id_; }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns nullptr if the device is
  // not local to this host.
  LocalDeviceState* local_device_state() const {
    return local_device_state_.get();
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns an error if the device
  // is not local to this host.
  StatusOr<LocalDeviceState*> GetLocalDeviceState() const;

  // The ID of this device's host. This is always 0 on single-host platforms.
  int host_id() const { return host_id_; }

  const std::string& platform_name() const { return platform_name_; }

  virtual std::string DebugString() const = 0;

 private:
  const int id_;
  const std::unique_ptr<LocalDeviceState> local_device_state_;
  const int host_id_;
  const std::string platform_name_;
};

class CpuDevice : public Device {
 public:
  using Device::Device;
  std::string DebugString() const override;
};

class GpuDevice : public Device {
 public:
  using Device::Device;
  std::string DebugString() const override;
};

struct AllocatorConfig {
  enum class Kind {
    kDefault,   // Client picks the best option for the platform.
    kPlatform,  // The platform's default.
    kBFC,  // Allocator using a "Best-Fit with Coalescing" algorithm. Currently
           // only available for GPU.
  };
  Kind kind = Kind::kDefault;

  // Only used if kind == kBFC. The maximum fraction of available memory to
  // allocate.
  double memory_fraction = 0.9;

  // Only used if kind == kBFC. If true, the allocator will immediately allocate
  // the maximum amount allowed by `memory_fraction`. This reduces
  // fragmentation, allowing more of the total memory to be used. If false, the
  // allocator will allocate more memory as allocations are requested.
  bool preallocate = true;
};

// Encapsulates the state of Python session with XLA.
class PyLocalClient {
 public:
  // Initializes a local XLA client for `platform_name`. Returns an error if no
  // such platform exists, or if the platform has no visible devices.
  static StatusOr<std::shared_ptr<PyLocalClient>> Get(
      const std::string& platform_name, const std::string& xla_platform_name,
      bool asynchronous, const AllocatorConfig& allocator_config);

  // `allocator` may null, in which case the platform default allocator is used.
  explicit PyLocalClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::shared_ptr<Device>> devices, int host_id,
      std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      std::unique_ptr<tensorflow::Allocator> host_memory_allocator);
  virtual ~PyLocalClient() = default;

  Status TransferToInfeed(const LiteralSlice& literal,
                          std::shared_ptr<Device> device);
  StatusOr<Literal> TransferFromOutfeed(const Shape& shape,
                                        std::shared_ptr<Device> device);

  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const;

  int device_count() const { return devices_.size(); }
  int local_device_count() const { return local_devices_.size(); }
  const std::vector<std::shared_ptr<Device>>& devices() const {
    return devices_;
  }
  const std::vector<std::shared_ptr<Device>>& local_devices() const {
    return local_devices_;
  }
  const std::map<int, std::shared_ptr<Device>>& id_to_device() const {
    return id_to_device_;
  }
  int host_id() const { return host_id_; }
  const std::string& platform_name() const { return platform_name_; }

  LocalDeviceState& device_state(int device_ordinal) const {
    return *local_devices_.at(device_ordinal)->local_device_state();
  }

  LocalClient* client() const { return client_; }
  se::DeviceMemoryAllocator* allocator() const { return allocator_; }
  tensorflow::Allocator* host_memory_allocator() const {
    return host_memory_allocator_.get();
  }

  tensorflow::thread::ThreadPool* h2d_transfer_pool() {
    return &h2d_transfer_pool_;
  }

  // Most platforms expect device-to-device transfers to be enqueued on the
  // source d2d stream, but some platforms use the destination d2d stream. This
  // function specifies which one the platform expects.
  virtual bool EnqueueD2DTransfersOnSrcStream() const { return true; }

 protected:
  std::string platform_name_;
  LocalClient* client_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::shared_ptr<Device>> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, std::shared_ptr<Device>> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<std::shared_ptr<Device>> local_devices_;
  int host_id_;

  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;

  // Allocator to be used for staging memory transfers to devices. Optional;
  // only used on GPU where it is more efficient to copy buffers to and from the
  // device via a staging area of pinned memory.
  std::unique_ptr<tensorflow::Allocator> host_memory_allocator_;

  tensorflow::thread::ThreadPool h2d_transfer_pool_;
};

// Holds a reference from Python to one or more device buffers.
// A PyLocalBuffer can be either valid or invalid. An invalid buffer is one that
// has never been initialized, or a buffer that has been deleted (e.g., by
// calling Delete). We allow PyLocalBuffer objects to outlive the underlying
// device buffers so we can decouple buffer lifetimes from the corresponding
// Python references if needed.
// Thread-safe.
class PyLocalBuffer {
 public:
  static StatusOr<std::unique_ptr<PyLocalBuffer>> FromLiterals(
      std::vector<BorrowingLiteral> leaves_literals, const Shape& tuple_shape,
      std::shared_ptr<void> leaves_reference,
      std::shared_ptr<PyLocalClient> client, std::shared_ptr<Device> device);

  static StatusOr<std::unique_ptr<PyLocalBuffer>> MakeTuple(
      const std::vector<PyLocalBuffer*> buffers,
      std::shared_ptr<PyLocalClient> client, std::shared_ptr<Device> device);

  PyLocalBuffer(Shape on_host_shape, Shape on_device_shape,
                std::shared_ptr<SharedDeviceBuffer> device_buffer,
                std::shared_ptr<PyLocalClient> client,
                std::shared_ptr<Device> device);

  PyLocalBuffer(const PyLocalBuffer&) = delete;
  PyLocalBuffer(PyLocalBuffer&&) = delete;
  PyLocalBuffer& operator=(const PyLocalBuffer&) = delete;
  PyLocalBuffer& operator=(PyLocalBuffer&&) = delete;

  const Shape& on_host_shape() const { return on_host_shape_; }
  const Shape& on_device_shape() const { return on_device_shape_; }
  std::shared_ptr<Device> device() const { return device_; }
  const std::string& platform_name() const { return client_->platform_name(); }
  std::shared_ptr<PyLocalClient> client() const { return client_; }

  // Returns the buffer's value as a tuple DAG of Python arrays. If the value
  // has previously been prefetched to the host, then returns the prefetched
  // version, otherwise copies the buffer to the host. Blocks until the
  // value is ready.
  StatusOr<std::shared_ptr<Literal>> ToLiteral();

  // Initiates a copy of the buffer to the host. Does not block waiting for
  // the transfer to complete. The value can be retrieved by a later call to
  // ToLiteral().
  Status CopyToHostAsync();

  // Returns the associated device buffer. Returns a nullptr if the buffer is
  // invalid.
  std::shared_ptr<SharedDeviceBuffer> DeviceBuffer() const;

  // Deletes the device memory associated with this buffer, leaving it in an
  // invalid state.
  void Delete();

  // Returns a view of the PyLocalBuffer DAG as a ShapedBuffer. The
  // PyLocalBuffer retains ownership of the device buffers.
  StatusOr<ShapedBuffer> AsShapedBuffer() const;

  // Destructures a tuple-valued PyLocalBuffer into its constituent elements.
  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> DestructureTuple();

  // Copies the buffer to device `dst_device`.
  StatusOr<std::unique_ptr<PyLocalBuffer>> CopyToDevice(
      std::shared_ptr<Device> dst_device);

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  Status BlockHostUntilReady();

 private:
  const std::shared_ptr<PyLocalClient> client_;
  const Shape on_host_shape_;
  const Shape on_device_shape_;
  const std::shared_ptr<Device> device_;
  mutable absl::Mutex mu_;
  std::shared_ptr<SharedDeviceBuffer> device_buffer_ GUARDED_BY(mu_);

  // The cached value of the buffer on the host, produced either from a call to
  // CopyToHost or from a call to ToLiteral. Once a value has been fetched to
  // the host, it persists Delete() is called or the PyLocalBuffer is destroyed.
  struct HostValue {
    absl::Notification ready;
    // status and value are valid for reading only after `ready` has been
    // notified.
    Status status;
    std::shared_ptr<Literal> value;
  };
  std::shared_ptr<HostValue> host_value_ GUARDED_BY(mu_);
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Wraps an XLA LocalExecutable.
class PyLocalExecutable {
 public:
  // Compiles a computation to an executable.
  static StatusOr<std::unique_ptr<PyLocalExecutable>> CompileForDevices(
      const XlaComputation& computation,
      absl::optional<std::vector<Shape>> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyLocalClient> client,
      const std::vector<std::vector<std::shared_ptr<Device>>>&
          device_assignment);

  // TODO(phawkins): Deprecated. Delete once all callers have been updated to
  // use the newer form.
  static StatusOr<std::unique_ptr<PyLocalExecutable>> Compile(
      const XlaComputation& computation,
      absl::optional<std::vector<Shape>> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyLocalClient> client,
      absl::optional<DeviceAssignment> device_assignment);

  PyLocalExecutable(std::shared_ptr<LocalExecutable> executable,
                    DeviceAssignment device_assignment,
                    std::shared_ptr<PyLocalClient> client);

  int num_replicas() const {
    return executable_->build_options().num_replicas();
  }

  int num_partitions() const {
    return executable_->build_options().num_partitions();
  }

  int64 SizeOfGeneratedCodeInBytes() const {
    return executable_->executable()->SizeOfGeneratedCodeInBytes();
  }

  const DeviceAssignment& device_assignment() const {
    return *device_assignment_;
  }

  const std::vector<std::shared_ptr<Device>>& local_devices() const {
    return local_devices_;
  }

  StatusOr<std::unique_ptr<PyLocalBuffer>> Execute(
      absl::Span<PyLocalBuffer* const> argument_handles);

  // Execute on many replicas. Takes a sequence of argument lists (one argument
  // list per replica) and returns a tuple of results (one result per replica).
  // The number of argument lists must be equal to the replica count.
  // The executable must have only one partition.
  // TODO(cjfj): Remove this once JAX is moved to `ExecuteOnLocalDevices`.
  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> ExecutePerReplica(
      absl::Span<const std::vector<PyLocalBuffer*>> argument_handles);

  // Execute on local devices. Takes a sequence of argument lists (one argument
  // list per local device) and returns a tuple of results (one result per local
  // device). The number of argument lists must be equal to the local device
  // count.
  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> ExecuteOnLocalDevices(
      absl::Span<const std::vector<PyLocalBuffer*>> argument_handles);

  void Delete() { executable_ = nullptr; }

  const string& name() const;

 private:
  StatusOr<std::unique_ptr<PyLocalBuffer>> ExecuteHelper(
      absl::Span<PyLocalBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id);

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  std::shared_ptr<PyLocalClient> const client_;
  std::shared_ptr<LocalExecutable> executable_;
  std::shared_ptr<DeviceAssignment> device_assignment_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. local_logical_devices_[i] = (i, 0)), but this may not be the case on
  // multi-host platforms.
  // If there are 4 replicas and 2 partitions on a single host platform, size of
  // local_logical_devices_ is 4*2 = 8.
  std::vector<std::pair<int, int>> local_logical_devices_;

  // local_devices_[i] is the Device to which local_logical_devices_[i] is
  // assigned.
  // shared_ptrs instead of unique_ptrs to play well with the Python bindings
  // (see xla.cc).
  std::vector<std::shared_ptr<Device>> local_devices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_CLIENT_H_
