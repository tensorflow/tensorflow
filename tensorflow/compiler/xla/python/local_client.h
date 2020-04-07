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

#include "absl/container/inlined_vector.h"
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
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/thread_annotations.h"

// API notes:
// Despite having the name "PyLocalClient", it is intended that this API may
// also be consumed from C++. Python/pybind11/NumPy logic should therefore not
// be used in this API.

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

  virtual std::string DebugString() const;

 private:
  const int id_;
  const std::unique_ptr<LocalDeviceState> local_device_state_;
  const int host_id_;
  const std::string platform_name_;
};

// Forward declaration.
class PyLocalBuffer;

// Helper struct for cross host transfers, returned by the callback from a call
// to PyLocalBuffer::MakeCrossHostReceiveBuffers.
struct PyLocalCrossHostRecvBuffer {
  // serialized_descriptor should be transmitted to the sender and passed to a
  // call to src_buffer->CopyToRemoteDevice.
  std::string serialized_descriptor;
  // The buffer that will hold the result of the transfer.
  std::unique_ptr<PyLocalBuffer> buffer;
};
using PyLocalCrossHostRecvNotifier =
    std::function<void(StatusOr<std::vector<PyLocalCrossHostRecvBuffer>>&&)>;

// Encapsulates the state of Python session with XLA.
//
// It is the responsibility of the client of this API to keep the PyLocalClient
// alive as long as any of the other runtime objects are alive.
class PyLocalClient : public std::enable_shared_from_this<PyLocalClient> {
 public:
  // `allocator` may null, in which case the platform default allocator is used.
  explicit PyLocalClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<Device>> devices, int host_id,
      std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
      std::unique_ptr<GpuExecutableRunOptions> gpu_run_options);
  virtual ~PyLocalClient() = default;

  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const;

  int device_count() const { return devices_.size(); }
  int local_device_count() const { return local_devices_.size(); }
  const std::vector<std::unique_ptr<Device>>& devices() const {
    return devices_;
  }
  const std::vector<Device*>& local_devices() const { return local_devices_; }
  const std::map<int, Device*>& id_to_device() const { return id_to_device_; }
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

  GpuExecutableRunOptions* gpu_run_options() const {
    return gpu_run_options_.get();
  }

  tensorflow::thread::ThreadPool* h2d_transfer_pool() {
    return &h2d_transfer_pool_;
  }

  // Most platforms expect device-to-device transfers to be enqueued on the
  // source d2d stream, but some platforms use the destination d2d stream. This
  // function specifies which one the platform expects.
  virtual bool EnqueueD2DTransfersOnSrcStream() const { return true; }

 protected:
  friend class PyLocalBuffer;
  virtual void EnqueueCrossHostReceive(
      std::vector<std::unique_ptr<PyLocalBuffer>>&& buffers,
      PyLocalCrossHostRecvNotifier&& notifier) const {
    notifier(Unimplemented("Cross host receives not implemented."));
  }

  virtual Status CopyToRemoteDevice(
      PyLocalBuffer* buffer, absl::string_view serialized_descriptor) const {
    return Unimplemented("Cross host sends not implemented.");
  }

  std::string platform_name_;
  LocalClient* client_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::unique_ptr<Device>> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, Device*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<Device*> local_devices_;
  int host_id_;

  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;

  // Allocator to be used for staging memory transfers to devices. Optional;
  // only used on GPU where it is more efficient to copy buffers to and from the
  // device via a staging area of pinned memory.
  std::unique_ptr<tensorflow::Allocator> host_memory_allocator_;

  std::unique_ptr<GpuExecutableRunOptions> gpu_run_options_;

  tensorflow::thread::ThreadPool h2d_transfer_pool_;
};

// Converts a 2D set of Device objects indexed by [replica][partition] into an
// xla::DeviceAssignment.
StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
    absl::Span<const std::vector<Device*>> devices);

// Holds a reference from Python to a tuple of device buffers. A PyLocalBuffer
// can be either valid or invalid. An invalid buffer is one that has never been
// initialized, or a buffer that has been deleted (e.g., by calling Delete). We
// allow PyLocalBuffer objects to outlive the underlying device buffers so we
// can decouple buffer lifetimes from the corresponding Python references if
// needed. Thread-safe.
class PyLocalBuffer {
 public:
  // Helper class to retain a "hold" on a PyLocalBuffer. A ScopedHold may not
  // outlive its parent PyLocalBuffer.
  //
  // There are two types of hold, as follows:
  //
  // 1) Usage hold: a transient hold while an operation using the buffer is
  //    being enqueued onto a stream.
  // A client acquires a usage hold by calling
  // PyLocalBuffer::GetBufferWithHold(kUsage) or the convenience wrapper
  // GetBufferWithUsageHold(). If the enqueue completes successfully the hold
  // should be released using a call to ConvertUsageHold. If the ScopedHold is
  // deleted without ConvertUsageHold being called, e.g., on error, the hold is
  // dropped. It is legal to drop a usage hold instead of calling
  // ConvertUsageHold, even if the buffer was successfully enqueued, as long as
  // the client ensures that all necessary synchronization has been done.
  //
  // 2) External hold: a potentially long-lived hold while the buffer is being
  //    shared by an external framework, e.g., NumPy.
  // A client acquires an external hold by calling
  // PyLocalBuffer::GetBufferWithHold(kExternal) or the convenience wrapper
  // GetBufferWithExternalReference and releases it by deleting the ScopedHold.
  // The external framework should not modify the underlying buffer unless it is
  // confident via its own synchronization that modifications do not race with
  // reads from the PyLocalBuffer.
  //
  // Calls to PyLocalBuffer::Release (and transitively to
  // PyLocalBuffer::Delete() and ~PyLocalBuffer()) will block until all usage
  // holds are either deleted or converted.
  class ScopedHold {
   public:
    enum Type { kUsage = 0, KExternalReference, kMaxValue };

    ~ScopedHold();
    ScopedHold(ScopedHold&& other);
    ScopedHold(const ScopedHold&) = delete;
    ScopedHold& operator=(const ScopedHold&) = delete;

    Type type() const { return type_; }

    Status status() const { return buffer_or_.status(); }
    bool ok() const { return buffer_or_.ok(); }

    // Access to the underlying device buffer storage. Requires this->ok().
    const std::shared_ptr<SharedDeviceBuffer>& buffer() const {
      CHECK_NE(buffer_or_.ValueOrDie(), nullptr);
      return buffer_or_.ValueOrDie();
    }
    SharedDeviceBuffer* operator->() const { return buffer().get(); }
    const SharedDeviceBuffer& operator*() const { return *buffer(); }

    // Converts the hold into a usage event. Only valid for holds of type
    // kUsage.
    //
    //   usage_stream:   the stream that the buffer was used on.
    //   event:          an event that has been recorded on usage_stream after
    //                   the buffer was used.
    //   reference_held: true if and only if the caller has caused a
    //                   reference to this->buffer() to stay live until after
    //                   the host is sure that the usage (transfer or execution)
    //                   has completed.
    void ConvertUsageHold(se::Stream* usage_stream,
                          std::shared_ptr<BufferDefinitionEvent> event,
                          bool reference_held);

   private:
    friend class PyLocalBuffer;

    // Helper struct that makes it possible to move a ScopedHold through a
    // closure.
    using ForClosure =
        std::tuple<PyLocalBuffer*, Type,
                   StatusOr<std::shared_ptr<SharedDeviceBuffer>>>;

    ScopedHold(PyLocalBuffer* parent, Type type)
        : parent_(parent), type_(type) {
      SetError(InvalidArgument("Buffer has not been initialized"));
    }
    explicit ScopedHold(const ForClosure& closure_helper)
        : parent_(std::get<0>(closure_helper)),
          type_(std::get<1>(closure_helper)),
          buffer_or_(std::get<2>(closure_helper)) {
      // Check the buffer is not in an error state.
      CHECK(buffer_or_.ValueOrDie() != nullptr);
    }

    // Sets error status.
    void SetError(Status s) { buffer_or_ = s; }

    // Sets buffer_or_. Called by parent_ to initialize the hold.
    void Acquire(StatusOr<std::shared_ptr<SharedDeviceBuffer>>&& buffer_or);
    // Releases the contents of *this, so *this can subsequently be
    // deleted without releasing the parent's hold. Should be passed to the
    // appropriate constructor of another ScopedHold, e.g., when a hold must be
    // passed through a closure that is incompatible with std::move.
    ForClosure ToClosure();

    PyLocalBuffer* const parent_;
    const Type type_;
    // There is an invariant that if buffer_or_.ok() then
    // buffer_or_.ValueOrDie() != nullptr.
    StatusOr<std::shared_ptr<SharedDeviceBuffer>> buffer_or_;
  };

  // If `force_copy` is true, forces a copy of the input buffer on CPU.
  // Otherwise the library is free to alias the output buffer with `data`.
  // `buffer_reference` is an optional shared pointer that should be kept alive
  // by the runtime as long as the contents of `data` may still be accessed by
  // the runtime (may be nullptr).
  static StatusOr<std::unique_ptr<PyLocalBuffer>> FromHostBuffer(
      const void* data, const Shape& shape, bool force_copy,
      std::shared_ptr<void> buffer_reference, PyLocalClient* client,
      Device* device);

  static StatusOr<std::unique_ptr<PyLocalBuffer>> FromHostLiteral(
      const LiteralSlice& literal, PyLocalClient* client, Device* device);

  // Asynchronously makes a vector of PyLocalBuffers that can be used to receive
  // cross host transfers using `client` on `device'. `shapes` must be the exact
  // shapes, with identical layouts, corresponding to the buffers that will be
  // sent. When resources for the transfer are available, notifier will be
  // called with a vector of PyLocalCrossHostRecvBuffer structs, one for each
  // shape in `shapes`. Each struct contains a buffer that will contain the
  // received value, and an opaque string that should be transmitted to the
  // sending host and used in a call to CopyToRemoteDevice. None of the recv
  // buffers will become ready until *all* of the sends have completed.
  static void MakeCrossHostReceiveBuffers(
      absl::Span<const Shape> shapes, PyLocalClient* client, Device* device,
      PyLocalCrossHostRecvNotifier&& notifier);

  PyLocalBuffer(Shape on_host_shape, Shape on_device_shape,
                std::shared_ptr<SharedDeviceBuffer> device_buffer,
                PyLocalClient* client, Device* device);
  ~PyLocalBuffer();

  PyLocalBuffer(const PyLocalBuffer&) = delete;
  PyLocalBuffer(PyLocalBuffer&&) = delete;
  PyLocalBuffer& operator=(const PyLocalBuffer&) = delete;
  PyLocalBuffer& operator=(PyLocalBuffer&&) = delete;

  const Shape& on_host_shape() const { return on_host_shape_; }
  const Shape& on_device_shape() const { return on_device_shape_; }
  Device* device() const { return device_; }
  const std::string& platform_name() const { return client_->platform_name(); }
  PyLocalClient* client() const { return client_; }
  bool IsEmptyTuple() const {
    return on_host_shape_.IsTuple() && on_host_shape_.tuple_shapes_size() == 0;
  }

  // Returns the buffer's value as an XLA Literal. If the value has previously
  // been prefetched to the host, then returns the prefetched version, otherwise
  // copies the buffer to the host. Blocks until the value is ready.
  StatusOr<std::shared_ptr<Literal>> ToLiteral();

  // Initiates a copy of the buffer to the host. Does not block waiting for
  // the transfer to complete. The value can be retrieved by a later call to
  // ToLiteral().
  Status CopyToHostAsync();

  // Drops the buffer's reference to its associated device memory, leaving the
  // buffer in an invalid state. The memory will be freed lazily when all async
  // operations using the buffer have completed, according to the allocation
  // semantics of the underlying platform. Delete may briefly block if another
  // thread is in the process of enqueuing an operation on this buffer, but it
  // will never block for a stream operation to complete. If an external
  // framework holds a reference to the SharedDeviceBuffer via
  // GetBufferWithExternalReference, the memory will not be freed until the
  // external framework drops the reference.
  void Delete();

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // SharedDeviceBuffer rather than freeing the device memory, so that another
  // framework can take ownership of it. The buffer returned from Release may
  // be safely dropped at any time even if it still has pending async
  // operations. The client should call BlockHostUntilReady before calling
  // Release with wait_for_operations_to_complete=false, to ensure that the host
  // has synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  StatusOr<std::shared_ptr<SharedDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  // True if and only if Delete or Release has previously been called.
  bool IsDeleted();

  // Returns a view of the PyLocalBuffer device memory as a ShapedBuffer. The
  // PyLocalBuffer retains ownership of the device buffers.
  StatusOr<ShapedBuffer> AsShapedBuffer() const;

  // Returns a hold on the SharedDeviceBuffer holding the device
  // buffers. See comment on ScopedHold.
  ScopedHold GetBufferWithHold(ScopedHold::Type type);
  ScopedHold GetBufferWithUsageHold() {
    return GetBufferWithHold(ScopedHold::kUsage);
  }
  ScopedHold GetBufferWithExternalReference() {
    return GetBufferWithHold(ScopedHold::KExternalReference);
  }

  // Copies the buffer to device `dst_device`. Returns an error if the buffer is
  // already on dst_device.
  StatusOr<std::unique_ptr<PyLocalBuffer>> CopyToDevice(Device* dst_device);

  // Copies the buffer to the remote device encoded in serialized_descriptor.
  // This call must be preceded by a call to MakeCrossHostReceiveBuffers on the
  // remote host's destination device. MakeCrossHostReceiveBuffers takes an
  // array of shapes to construct the destination buffers, and a callback
  // supplies an array containing both the destination buffers, and a serialized
  // descriptor for each buffer. For each destination buffer there should be a
  // matching call to src->CopyToRemoteDevice on a remote host for a src buffer
  // of the corresponding shape. serialized_descriptor is the string returned by
  // the callback along with the corresponding destination buffer.
  Status CopyToRemoteDevice(absl::string_view serialized_descriptor);

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  Status BlockHostUntilReady();

 private:
  friend class PyLocalClient;
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

  // If device_buffer_ is non-null, adds a hold of 'type' and returns
  // device_buffer_. Otherwise returns an error.
  StatusOr<std::shared_ptr<SharedDeviceBuffer>> GetBufferForHoldLocked(
      ScopedHold::Type type) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // If device_buffer_ is non-null, adds a hold of hold->type() and initializes
  // `hold` with device_buffer_. Otherwise initializes `hold` with an error
  // status.
  void AcquireHoldLocked(ScopedHold* hold) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Drops a usage hold and calls device_buffer_->AddUsageEvent. Does a sanity
  // check that buffer==device_buffer_ or device_buffer_==nullptr. Called after
  // device_buffer_ was successfully enqueued on a stream.
  void ConvertUsageHold(SharedDeviceBuffer* buffer, se::Stream* usage_stream,
                        std::shared_ptr<BufferDefinitionEvent> event,
                        bool reference_held);

  // Drops a hold without taking any other action. Does a sanity check that
  // buffer==device_buffer_ or device_buffer_==nullptr.
  void DropHold(ScopedHold::Type type, SharedDeviceBuffer* buffer);

  StatusOr<std::pair<std::unique_ptr<PyLocalBuffer>,
                     std::shared_ptr<BufferDefinitionEvent>>>
  CopyToDeviceHelper(Device* dst_device, LocalDeviceState* dst_local_device,
                     LocalDeviceState* transfer_local_device,
                     se::Stream* transfer_stream,
                     std::shared_ptr<SharedDeviceBuffer> src_device_buffer);

  PyLocalClient* const client_;
  const Shape on_host_shape_;
  const Shape on_device_shape_;
  Device* const device_;

  mutable absl::Mutex mu_;
  std::shared_ptr<SharedDeviceBuffer> device_buffer_ TF_GUARDED_BY(mu_);
  std::shared_ptr<HostValue> host_value_ TF_GUARDED_BY(mu_);
  // Count of holds on the buffer.
  std::array<int, ScopedHold::Type::kMaxValue> holds_ TF_GUARDED_BY(mu_);
};

struct CompileOptions {
  // The layouts of the arguments that the computation should expect.
  absl::optional<std::vector<Shape>> argument_layouts;

  // If true, the arguments to the computation will be wrapped in a tuple and
  // passed as a single parameter.
  bool tuple_arguments = false;

  // XLA's compilation time options.
  ExecutableBuildOptions executable_build_options;
};

struct ExecuteOptions {
  // If true, the computation must return a tuple, which will be destructured
  // into its elements.
  bool untuple_result = false;
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Wraps one or more XLA LocalExecutables (one per
// partition, as specified by the build options).
class PyLocalExecutable {
 public:
  static StatusOr<std::unique_ptr<PyLocalExecutable>> Compile(
      const XlaComputation& computation, PyLocalClient* client,
      CompileOptions options);

  PyLocalExecutable(std::vector<std::unique_ptr<LocalExecutable>> executables,
                    bool tuple_arguments, DeviceAssignment device_assignment,
                    PyLocalClient* client);

  PyLocalClient* client() const { return client_; }

  int num_replicas() const {
    return executables_[0]->build_options().num_replicas();
  }

  int num_partitions() const {
    return executables_[0]->build_options().num_partitions();
  }

  int64 SizeOfGeneratedCodeInBytes() const {
    int64 size = 0;
    for (auto& executable : executables_) {
      size += executable->executable()->SizeOfGeneratedCodeInBytes();
    }
    return size;
  }

  const std::vector<std::shared_ptr<LocalExecutable>>& executables() const {
    return executables_;
  }

  const DeviceAssignment& device_assignment() const {
    return *device_assignment_;
  }

  const std::vector<std::pair<int, int>>& local_logical_device_ids() const {
    return local_logical_device_ids_;
  }

  const std::vector<Device*>& local_devices() const { return local_devices_; }

  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> Execute(
      absl::Span<PyLocalBuffer* const> argument_handles,
      const ExecuteOptions& options) const;

  // Execute on local devices. Takes a sequence of argument lists (one argument
  // list per local device) and returns a tuple of results (one result per local
  // device). The number of argument lists must be equal to the local device
  // count.
  StatusOr<std::vector<std::vector<std::unique_ptr<PyLocalBuffer>>>>
  ExecuteOnLocalDevices(
      absl::Span<const std::vector<PyLocalBuffer*>> argument_handles,
      const ExecuteOptions& options) const;

  void Delete() { executables_.clear(); }

  const string& name() const;

 private:
  StatusOr<ScopedShapedBuffer> EnqueueExecution(
      absl::Span<PyLocalBuffer* const> argument_handles, int replica,
      int partition, int executable_idx, const RunId& run_id,
      const ExecuteOptions& options, Device* device,
      std::vector<PyLocalBuffer::ScopedHold>* device_buffers) const;
  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> ExecuteHelper(
      absl::Span<PyLocalBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options) const;

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  PyLocalClient* const client_;
  // One executable per partition.
  std::vector<std::shared_ptr<LocalExecutable>> executables_;
  std::shared_ptr<DeviceAssignment> device_assignment_;

  // True if the executables were compiled expecting arguments in a single
  // tuple.
  const bool tuple_arguments_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. local_logical_device_ids_[i] = (i, 0)), but this may not be the case
  // on multi-host platforms.
  // If there are 4 replicas and 2 partitions on a single host platform, size of
  // local_logical_device_ids_ is 4*2 = 8.
  std::vector<std::pair<int, int>> local_logical_device_ids_;

  // local_devices_[i] is the Device to which local_logical_device_ids_[i] is
  // assigned.
  // shared_ptrs instead of unique_ptrs to play well with the Python bindings
  // (see xla.cc).
  std::vector<Device*> local_devices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_CLIENT_H_
