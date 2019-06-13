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

#include <deque>
#include <string>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/shared_device_buffer.h"
#include "tensorflow/compiler/xla/python/worker_thread.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Registers a 'fn_capsule' as a CPU custom call target.
// 'fn_capsule' is a void* pointer encapsulated in a PyCapsule object, with name
// "xla._CPU_CUSTOM_CALL_TARGET".
Status RegisterCpuCustomCallTarget(const std::string& fn_name,
                                   pybind11::capsule capsule);

// Class that encapsulates state relating to a device (e.g., a GPU) on which we
// can perform computation and transfers.
class Device {
 public:
  // If use_multiple_streams is true, we allocate separate streams for compute
  // and transfers. If it is false, we share a single stream for compute and
  // transfers. The CPU device does not support multiple streams, and this is
  // a workaround until it does.
  //
  // If synchronous_deallocation is true, the host must not free buffers until
  // compute/transfers that use those buffers have completed. For example, this
  // typically is the case for the "platform" where compute/transfers are
  // operations that take place on another thread.
  //
  // If asynchronous is false, the host will synchronize to the device after
  // each execution or transfer. This is intended for debugging only.
  Device(se::StreamExecutor* executor, bool use_multiple_streams,
         bool synchronous_deallocation, bool asynchronous);
  ~Device();

  bool use_multiple_streams() const { return use_multiple_streams_; }
  bool synchronous_deallocation() const { return synchronous_deallocation_; }
  bool asynchronous() const { return asynchronous_; }
  se::Stream* compute_stream() const { return compute_stream_.get(); }
  se::Stream* host_to_device_stream() const {
    return host_to_device_stream_.get();
  }
  se::Stream* device_to_host_stream() const {
    return device_to_host_stream_.get();
  }

  // A worker thread, used for replicated computation launches and callbacks.
  WorkerThread* worker_thread() const { return worker_thread_.get(); }

  // Enqueues a host callback on 'stream', to be executed by worker_thread_.
  // ThenDoHostCallback is often constrained in what it can do, in particular,
  // on GPU the callback runs on a thread belonging to the GPU runtime and
  // cannot perform GPU operations itself.
  void ThenExecuteOnWorkerThread(se::Stream* stream,
                                 std::function<void()> callback) const;

  // Helper for releasing values from a callback at the tail of a stream.
  // This is only permitted if object's destructor will not free any device
  // objects, since the callback may be called from a device thread pool on
  // GPU.
  template <typename T>
  void ThenRelease(se::Stream* stream, T object) const {
    if (callback_stream_.get() != stream) {
      callback_stream_->ThenWaitFor(stream);
    }
    callback_stream_->ThenDoHostCallback(
        std::bind([](T& object) { /* releases object */ }, std::move(object)));
  }

  // Helpers for releasing values on a worker thread at the tail of a stream on
  // a worker thread.
  template <typename T>
  void ThenReleaseOnWorkerThread(se::Stream* stream,
                                 std::shared_ptr<T> object) const {
    // We use a non-smart pointer here because we want to ensure that the worker
    // thread is the only callee of the shared_ptr destructor, and if we passed
    // object by lambda capture we have a race where the worker thread might
    // run and release its reference first.
    auto* ref = new std::shared_ptr<T>(std::move(object));
    if (callback_stream_.get() != stream) {
      callback_stream_->ThenWaitFor(stream);
    }
    ThenExecuteOnWorkerThread(callback_stream_.get(), [ref]() { delete ref; });
  }
  template <typename T>
  void ThenReleaseOnWorkerThread(se::Stream* stream,
                                 std::vector<std::shared_ptr<T>> object) const {
    auto* ref = new std::vector<std::shared_ptr<T>>(std::move(object));
    if (callback_stream_.get() != stream) {
      callback_stream_->ThenWaitFor(stream);
    }
    ThenExecuteOnWorkerThread(callback_stream_.get(), [ref]() { delete ref; });
  }

 private:
  Status SynchronizeAllActivity();

  bool use_multiple_streams_;
  bool synchronous_deallocation_;
  bool asynchronous_;
  std::shared_ptr<se::Stream> compute_stream_;
  std::shared_ptr<se::Stream> host_to_device_stream_;
  std::shared_ptr<se::Stream> device_to_host_stream_;

  // Callback stream is used for running short host-side callbacks after device
  // side events, without preventing the device-side stream from doing useful
  // work.
  std::shared_ptr<se::Stream> callback_stream_;

  std::unique_ptr<WorkerThread> worker_thread_;
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
  double memory_fraction = 1.0;
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
  explicit PyLocalClient(std::string platform_name, LocalClient* client,
                         std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                         bool asynchronous);
  virtual ~PyLocalClient() = default;

  Status TransferToInfeed(const LiteralSlice& literal, int device_ordinal);
  StatusOr<pybind11::object> TransferFromOutfeed(const Shape& shape,
                                                 int device_ordinal);

  int device_count() const { return client_->device_count(); }
  const Device& device(int device_ordinal) const {
    return *devices_.at(device_ordinal);
  }
  LocalClient* client() const { return client_; }
  se::DeviceMemoryAllocator* allocator() const { return allocator_; }

  tensorflow::thread::ThreadPool* h2d_transfer_pool() {
    return &h2d_transfer_pool_;
  }

  PythonRefManager& py_ref_manager() { return py_ref_manager_; }

 protected:
  std::string platform_name_;
  LocalClient* client_;
  std::vector<std::unique_ptr<Device>> devices_;
  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;

  tensorflow::thread::ThreadPool h2d_transfer_pool_;

  PythonRefManager py_ref_manager_;
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
  static StatusOr<std::unique_ptr<PyLocalBuffer>> FromPython(
      const pybind11::object& argument, std::shared_ptr<PyLocalClient> client,
      int device_ordinal);

  // Converts multiple (python object, device ordinal) pairs into
  // PyLocalBuffers in parallel.
  static StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> FromPythonValues(
      const std::vector<std::pair<pybind11::object, int>>& argument,
      std::shared_ptr<PyLocalClient> client);

  static StatusOr<std::unique_ptr<PyLocalBuffer>> MakeTuple(
      const std::vector<PyLocalBuffer*> buffers,
      std::shared_ptr<PyLocalClient> client, int device_ordinal);

  PyLocalBuffer() = default;
  PyLocalBuffer(Shape on_host_shape,
                std::shared_ptr<PySharedDeviceBuffer> device_buffer,
                std::shared_ptr<PyLocalClient> client);

  PyLocalBuffer(const PyLocalBuffer&) = delete;
  PyLocalBuffer(PyLocalBuffer&&) = delete;
  PyLocalBuffer& operator=(const PyLocalBuffer&) = delete;
  PyLocalBuffer& operator=(PyLocalBuffer&&) = delete;

  const Shape& on_host_shape() const { return on_host_shape_; }
  int device_ordinal() const { return device_ordinal_; }

  // Returns the buffer's value as a tuple DAG of Python arrays. If the value
  // has previously been prefetched to the host, then returns the prefetched
  // version, otherwise copies the buffer to the host. Blocks until the
  // value is ready.
  StatusOr<pybind11::object> ToPython();

  // Initiates a copy of the buffer to the host. Does not block waiting for
  // the transfer to complete. The value can be retrieved by a later call to
  // ToPython().
  Status CopyToHostAsync();

  // Returns the associated device buffer. Returns a nullptr if the buffer is
  // invalid.
  std::shared_ptr<PySharedDeviceBuffer> DeviceBuffer() const;

  // Deletes the device memory associated with this buffer, leaving it in an
  // invalid state.
  void Delete();

  // Returns a view of the PyLocalBuffer DAG as a ShapedBuffer. The
  // PyLocalBuffer retains ownership of the device buffers.
  StatusOr<ShapedBuffer> AsShapedBuffer() const;

  // Destructures a tuple-valued PyLocalBuffer into its constituent elements.
  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> DestructureTuple();

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  Status BlockHostUntilReady();

 private:
  const std::shared_ptr<PyLocalClient> client_;
  const Shape on_host_shape_;
  const int device_ordinal_;
  mutable absl::Mutex mu_;
  std::shared_ptr<PySharedDeviceBuffer> device_buffer_ GUARDED_BY(mu_);

  // The cached value of the buffer on the host, produced either from a call to
  // CopyToHost or from a call to ToPython. Once a value has been fetched to
  // the host, it persists Delete() is called or the PyLocalBuffer is destroyed.
  struct HostValue {
    absl::Notification ready;
    // status and value are valid for reading only after `ready` has been
    // notified.
    Status status;
    std::shared_ptr<xla::Literal> value;
  };
  std::shared_ptr<HostValue> host_value_ GUARDED_BY(mu_);
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Wraps an XLA LocalExecutable.
class PyLocalExecutable {
 public:
  // Compiles a computation to an executable.
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

  // Returns the device ordinals to which each replica is assigned.
  std::vector<int> DeviceOrdinals() const;

  const DeviceAssignment& device_assignment() const {
    return device_assignment_;
  }

  StatusOr<std::unique_ptr<PyLocalBuffer>> Execute(
      absl::Span<PyLocalBuffer* const> argument_handles);

  // Execute on many replicas. Takes a sequence of argument lists (one argument
  // list per replica) and returns a tuple of results (one result per replica).
  // The number of argument lists must be equal to the replica count.
  StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>> ExecutePerReplica(
      absl::Span<const std::vector<PyLocalBuffer*>> argument_handles);

  void Delete() { executable_ = nullptr; }

 private:
  StatusOr<std::unique_ptr<PyLocalBuffer>> ExecuteHelper(
      absl::Span<PyLocalBuffer* const> argument_handles, int replica,
      const RunId& run_id);

  std::shared_ptr<PyLocalClient> const client_;
  std::shared_ptr<LocalExecutable> executable_;
  const DeviceAssignment device_assignment_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_CLIENT_H_
