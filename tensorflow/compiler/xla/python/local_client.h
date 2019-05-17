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

#include "absl/types/span.h"
#include "include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
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

// Class that manages destruction of Python objects.
//
// We must not destroy Python objects without holding the GIL. However, we
// frequently want to hold references to Python objects for the duration of
// an asynchronous transfer on a Stream, and release our reference when the
// transfer completes.
//
// This class holds references to Python objects outside a GIL scope, that can
// be collected later when the GIL is held by calling CollectGarbage().
class PythonRefManager {
 public:
  PythonRefManager() = default;

  // Holds references to a set of pybind11::objects, adding the references to
  // the PythonRefManager on destruction.
  class ManagedPyObjects {
   public:
    ManagedPyObjects() = default;
    ManagedPyObjects(PythonRefManager* manager,
                     absl::Span<pybind11::object> objects);

    ~ManagedPyObjects();

    ManagedPyObjects(const ManagedPyObjects& other) = default;
    ManagedPyObjects(ManagedPyObjects&& other) = default;
    ManagedPyObjects& operator=(const ManagedPyObjects& other) = default;
    ManagedPyObjects& operator=(ManagedPyObjects&& other) = default;

   private:
    PythonRefManager* manager_ = nullptr;
    absl::InlinedVector<pybind11::object, 1> objects_;
  };

  // Creates a managed std::shared_ptr to an object. When the shared_ptr is
  // destroyed, the reference to 'object' will be added to python_garbage_,
  // and collected next time CollectGarbage() is called.
  ManagedPyObjects ManageReferences(absl::Span<pybind11::object> objects);

  // Releases the contents of python_garbage_. Requires that the GIL is held.
  // The client calls this method during API entry points where the GIL is held
  // to free any garbage that has accumulated.
  void CollectGarbage();

 private:
  absl::Mutex mu_;
  std::deque<pybind11::object> python_garbage_ GUARDED_BY(mu_);
};

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

  // Only used if kind == kBFC. Fraction of available memory to allocate.
  double memory_fraction = .9;
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
class PyLocalBuffer {
 public:
  static StatusOr<PyLocalBuffer> FromPython(
      const pybind11::object& argument, std::shared_ptr<PyLocalClient> client,
      int device_ordinal);

  // Converts multiple (python object, device ordinal) pairs into
  // PyLocalBuffers in parallel.
  static StatusOr<std::vector<PyLocalBuffer>> FromPythonValues(
      const std::vector<std::pair<pybind11::object, int>>& argument,
      std::shared_ptr<PyLocalClient> client);

  static StatusOr<PyLocalBuffer> MakeTuple(
      const std::vector<PyLocalBuffer> buffers,
      std::shared_ptr<PyLocalClient> client, int device_ordinal);

  PyLocalBuffer() = default;
  PyLocalBuffer(Shape on_host_shape,
                std::shared_ptr<PySharedDeviceBuffer> device_buffer,
                std::shared_ptr<PyLocalClient> client);
  StatusOr<pybind11::object> ToPython() const;
  const Shape& on_host_shape() const { return on_host_shape_; }
  const std::shared_ptr<PySharedDeviceBuffer>& device_buffer() const {
    return device_buffer_;
  }
  int device_ordinal() const { return device_buffer_->device_ordinal(); }

  void Delete() {
    device_buffer_ = nullptr;
    client_ = nullptr;
  }

  // Returns a view of the PyLocalBuffer DAG as a ShapedBuffer. The
  // PyLocalBuffer retains ownership of the device buffers.
  ShapedBuffer AsShapedBuffer() const;

  // Destructures a tuple-valued PyLocalBuffer into its constituent elements.
  StatusOr<std::vector<PyLocalBuffer>> DestructureTuple();

 private:
  std::shared_ptr<PyLocalClient> client_ = nullptr;
  Shape on_host_shape_;
  std::shared_ptr<PySharedDeviceBuffer> device_buffer_;
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Wraps an XLA LocalExecutable.
class PyLocalExecutable {
 public:
  // Compiles a computation to an executable.
  static StatusOr<std::unique_ptr<PyLocalExecutable>> Compile(
      const XlaComputation& computation, std::vector<Shape> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyLocalClient> client);

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

  StatusOr<PyLocalBuffer> Execute(
      absl::Span<PyLocalBuffer* const> argument_handles);

  // Execute on many replicas. Takes a sequence of argument lists (one argument
  // list per replica) and returns a tuple of results (one result per replica).
  // The number of argument lists must be equal to the replica count.
  StatusOr<std::vector<PyLocalBuffer>> ExecutePerReplica(
      absl::Span<const std::vector<PyLocalBuffer*>> argument_handles);

  void Delete() { executable_ = nullptr; }

 private:
  StatusOr<PyLocalBuffer> ExecuteHelper(
      absl::Span<PyLocalBuffer* const> argument_handles, int replica);

  std::shared_ptr<PyLocalClient> const client_;
  std::shared_ptr<LocalExecutable> executable_;
  const DeviceAssignment device_assignment_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_LOCAL_CLIENT_H_
