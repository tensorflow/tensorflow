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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_TPU_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_TPU_CLIENT_H_

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/threadpool.h"

namespace xla {

inline const char* TpuPlatform() {
  static constexpr char kTpuPlatform[] = "tpu";
  return kTpuPlatform;
}

class TpuDeviceDescription : public PjRtDeviceDescription {
 public:
  TpuDeviceDescription(int id, int process_index,
                       const std::array<int, 3>& coords, int core_on_chip);

  const std::array<int, 3>& coords() const { return coords_; }
  int core_on_chip() const { return core_on_chip_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::string_view ToString() const override { return to_string_; }

  int id() const override { return id_; }

  int process_index() const override { return process_index_; }

  absl::string_view device_kind() const override { return device_kind_; }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  const int id_;
  const int process_index_;
  const std::array<int, 3> coords_;
  // Index of the core of the same chip.
  int core_on_chip_;
  const std::string device_kind_ = "Cloud TPU";
  std::string debug_string_;
  std::string to_string_;
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_ = {};
};

class PyTpuClient;

class TpuDevice : public PjRtDevice {
 public:
  TpuDevice(int id, int process_index, const std::array<int, 3>& coords,
            int core_on_chip);

  const TpuDeviceDescription& description() const override {
    return description_;
  }

  const std::array<int, 3>& coords() const { return description().coords(); }
  int core_on_chip() const { return description().core_on_chip(); }

  static xla::StatusOr<std::vector<std::shared_ptr<xla::PjRtDevice>>>
  GetTpuDevices(const tpu_driver::SystemInfo& system_info);

  PjRtClient* client() const override { return nullptr; }
  PyTpuClient* tpu_client() const { return tpu_client_; }
  void set_tpu_client(PyTpuClient* tpu_client) { tpu_client_ = tpu_client; }

  bool IsAddressable() const override { return false; }

  int local_hardware_id() const override { return -1; }

  Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("Infeed not yet implemented via this API");
  }

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented("Outfeed not yet implemented via this API");
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

 private:
  TpuDeviceDescription description_;
  PyTpuClient* tpu_client_;
};

// Encapsulates the state of Python session with XLA.
class PyTpuClient : public std::enable_shared_from_this<PyTpuClient> {
 public:
  // Initializes a local XLA client for `platform_name`. Returns an error if no
  // such platform exists, or if the platform has no visible devices.
  static StatusOr<std::shared_ptr<PyTpuClient>> Get(const std::string& worker);

  explicit PyTpuClient(std::string platform_name,
                       std::unique_ptr<tpu_driver::TpuDriver> driver,
                       std::vector<std::shared_ptr<PjRtDevice>> devices,
                       int process_index);
  virtual ~PyTpuClient() = default;

  PyTpuClient(const PyTpuClient&) = delete;
  PyTpuClient(PyTpuClient&&) = delete;
  PyTpuClient& operator=(const PyTpuClient&) = delete;
  PyTpuClient& operator=(PyTpuClient&&) = delete;

  Status TransferToInfeed(const LiteralSlice& literal, int device_id);
  StatusOr<Literal> TransferFromOutfeed(const Shape& shape, int device_id);

  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const;

  int device_count() const { return devices_.size(); }
  int local_device_count() const { return local_devices_.size(); }
  const std::vector<std::shared_ptr<PjRtDevice>>& devices() { return devices_; }
  const std::vector<std::shared_ptr<PjRtDevice>>& local_devices() {
    return local_devices_;
  }
  const std::map<int, std::shared_ptr<PjRtDevice>>& id_to_device() const {
    return id_to_device_;
  }
  int process_index() const { return process_index_; }
  const absl::string_view platform_name() const { return platform_name_; }
  const absl::string_view platform_version() const { return platform_version_; }

  StatusOr<Shape> ChooseCompactLayoutForShape(Shape subshape) {
    return Unimplemented("ChooseCompactLayoutForShape not implemented.");
  }

  // Returns a bad status containing `caller_name` if `device_id` doesn't
  // correspond to a valid device at the POD-slice boundary.
  Status CheckDeviceId(int device_id, absl::string_view caller_name);

  tpu_driver::TpuDriver* driver() { return driver_.get(); }

  tsl::thread::ThreadPool* GetThreadPool() { return pool_.get(); }

 protected:
  std::string platform_name_;
  std::string platform_version_;
  std::unique_ptr<tpu_driver::TpuDriver> driver_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::shared_ptr<PjRtDevice>> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, std::shared_ptr<PjRtDevice>> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<std::shared_ptr<PjRtDevice>> local_devices_;
  int process_index_;

  // A thread pool for scheduling core executions in parallel.
  std::unique_ptr<tsl::thread::ThreadPool> pool_;
};

// Manages a buffer shared amongst multiple users. Buffers are asynchronously
// deallocated after the last use.
struct TpuSharedBuffer final {
 public:
  TpuSharedBuffer(tpu_driver::TpuDriver* driver,
                  std::unique_ptr<tpu_driver::BufferHandle> handle,
                  std::vector<std::shared_ptr<tpu_driver::Event>> wait_for_use,
                  std::shared_ptr<PjRtDevice> src_device)
      : driver(driver),
        device(std::move(src_device)),
        handle(std::move(handle)),
        wait_for_use(std::move(wait_for_use)) {}

  ~TpuSharedBuffer() {
    std::vector<tpu_driver::Event*> events;
    events.reserve(wait_for_use.size());
    for (const auto& e : wait_for_use) {
      events.push_back(e.get());
    }
    driver->Deallocate(std::move(handle), events);
  }

  tpu_driver::TpuDriver* const driver;
  const std::shared_ptr<PjRtDevice> device;

  std::unique_ptr<tpu_driver::BufferHandle> handle;
  std::vector<std::shared_ptr<tpu_driver::Event>> wait_for_use;
};

// Holds a reference from Python to one or more device buffers.
// A PyTpuBuffer can be either valid or invalid. An invalid buffer is one that
// has never been initialized, or a buffer that has been deleted (e.g., by
// calling Delete). We allow PyTpuBuffer objects to outlive the underlying
// device buffers so we can decouple buffer lifetimes from the corresponding
// Python references if needed.
// Thread-safe.
class PyTpuBuffer {
 public:
  // `tuple_shape` can be at most a one-level tuple combining non-tuple leaves.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> FromLiterals(
      std::vector<BorrowingLiteral> leaves_literals, const Shape& tuple_shape,
      std::shared_ptr<void> leaves_reference,
      std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device);

  // Supports nested tuple creation.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> MakeTuple(
      absl::Span<PyTpuBuffer* const> buffers,
      std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device);

  PyTpuBuffer() = delete;
  PyTpuBuffer(Shape on_host_shape,
              std::shared_ptr<TpuSharedBuffer> device_buffer,
              std::vector<std::shared_ptr<TpuSharedBuffer>> child_buffers,
              std::shared_ptr<PyTpuClient> client);

  PyTpuBuffer(const PyTpuBuffer&) = delete;
  PyTpuBuffer(PyTpuBuffer&&) = delete;
  PyTpuBuffer& operator=(const PyTpuBuffer&) = delete;
  PyTpuBuffer& operator=(PyTpuBuffer&&) = delete;

  const Shape& on_host_shape() const { return on_host_shape_; }
  std::shared_ptr<PjRtDevice> device() const { return device_; }
  const absl::string_view platform_name() const {
    return client_->platform_name();
  }
  std::shared_ptr<PyTpuClient> client() const { return client_; }

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
  std::shared_ptr<TpuSharedBuffer> DeviceBuffer() const;

  // Deletes the device memory associated with this buffer, leaving it in an
  // invalid state.
  void Delete();

  // Destructures a tuple-valued PyTpuBuffer into its constituent elements.
  StatusOr<std::vector<std::unique_ptr<PyTpuBuffer>>> DestructureTuple();

  // Copies the buffer to target device `dst_device` and returns a PyTpuBuffer
  // object holding the context to the target device buffer.
  StatusOr<std::unique_ptr<PyTpuBuffer>> CopyToDevice(
      std::shared_ptr<PjRtDevice> dst_device);

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  Status BlockHostUntilReady();

  // Allocates uninitialized buffers on device `device_id`. If `shape` is a
  // tuple, the returned buffer corresponds to the root tuple buffer.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> AllocateBuffer(
      const Shape& shape, std::shared_ptr<PyTpuClient> client,
      std::shared_ptr<PjRtDevice> device);

 private:
  // Initializes a just allocated device buffer. The returned event will be
  // placed into the buffer's `wait_for_use` list.
  using BufferInitializer = std::function<std::shared_ptr<tpu_driver::Event>(
      tpu_driver::BufferHandle*)>;
  // Allocates and optionally initializes a non-tuple buffer on the device.
  static StatusOr<std::unique_ptr<PyTpuBuffer>> CreateBuffer(
      const Shape& non_tuple_shape,
      std::optional<BufferInitializer> initializer,
      std::shared_ptr<PyTpuClient> client, std::shared_ptr<PjRtDevice> device);

  const std::shared_ptr<PyTpuClient> client_;
  const Shape on_host_shape_;
  const std::shared_ptr<PjRtDevice> device_;

  // If this is a tuple, `device_buffer_` stores the tuple buffer and
  // `child_buffers_` stores the child buffers; else, `device_buffer_` stores
  // the data content and `child_buffers_` is empty.
  mutable absl::Mutex mu_;
  std::shared_ptr<TpuSharedBuffer> device_buffer_ ABSL_GUARDED_BY(mu_);
  std::vector<std::shared_ptr<TpuSharedBuffer>> child_buffers_
      ABSL_GUARDED_BY(mu_);
  // The cached value of the buffer on the host, produced either from a call to
  // CopyToHost or from a call to ToLiteral. Once a value has been fetched to
  // the host, it persists Delete() is called or the PyTpuBuffer is destroyed.
  struct HostValue {
    absl::Mutex mutex;
    absl::Notification ready;
    int pending_ops;
    // status and value are valid for reading only after `ready` has been
    // notified.
    Status status;
    std::shared_ptr<Literal> value;
  };
  std::shared_ptr<HostValue> host_value_ ABSL_GUARDED_BY(mu_);
};

// A dummy token that is always ready. PyTpuExecutable::Execute() is blocking
// until the computation finishes.
class PyTpuToken {
 public:
  PyTpuToken() {}
  Status Await() { return OkStatus(); }
};

class PyShardedTpuToken {
 public:
  PyShardedTpuToken() {}
  Status Await() { return OkStatus(); }
  PyTpuToken GetPyToken(int i) { return PyTpuToken(); }
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. Wraps an XLA LocalExecutable.
class PyTpuExecutable {
 public:
  static StatusOr<std::unique_ptr<PyTpuExecutable>> Compile(
      const XlaComputation& computation,
      std::optional<std::vector<Shape>> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyTpuClient> client, bool tuple_arguments);

  static StatusOr<std::unique_ptr<PyTpuExecutable>> CompileMlir(
      mlir::ModuleOp module, std::optional<std::vector<Shape>> argument_layouts,
      const ExecutableBuildOptions* build_options,
      std::shared_ptr<PyTpuClient> client, bool tuple_arguments);

  PyTpuExecutable(
      std::unique_ptr<tpu_driver::CompiledProgramHandle> compiled_program,
      DeviceAssignment device_assignment, std::shared_ptr<PyTpuClient> client,
      xla::Shape result_shape, bool tuple_arguments);
  virtual ~PyTpuExecutable() {
    for (auto it = executables_.begin(); it != executables_.end(); ++it) {
      client_->driver()->UnloadProgram(std::move(it->second), {});
    }
  }

  PyTpuExecutable(const PyTpuExecutable&) = delete;
  PyTpuExecutable(PyTpuExecutable&&) = delete;
  PyTpuExecutable& operator=(const PyTpuExecutable&) = delete;
  PyTpuExecutable& operator=(PyTpuExecutable&&) = delete;

  std::shared_ptr<PyTpuClient> client() const { return client_; }

  int num_replicas() const { return device_assignment_.replica_count(); }
  int num_partitions() const { return device_assignment_.computation_count(); }

  int64_t SizeOfGeneratedCodeInBytes() const {
    CHECK_GE(executables_.size(), 1);
    return executables_.begin()->second->size_in_bytes();
  }

  const DeviceAssignment& device_assignment() const {
    return device_assignment_;
  }

  const std::vector<std::pair<int, int>>& local_logical_device_ids() const {
    return local_logical_device_ids_;
  }

  const std::vector<std::shared_ptr<PjRtDevice>>& local_devices() const {
    return local_devices_;
  }

  // TODO(power): Both Execute and ExecutePerOnLocalDevices block and wait
  // inside for computation to finish. Coordinate with JAX code change to see if
  // we can make both Execute and ExecutePerReplica non-blocking.
  StatusOr<std::vector<std::unique_ptr<PyTpuBuffer>>> Execute(
      absl::Span<PyTpuBuffer* const> argument_handles);

  StatusOr<std::pair<std::vector<std::unique_ptr<PyTpuBuffer>>, PyTpuToken>>
  ExecuteWithToken(absl::Span<PyTpuBuffer* const> argument_handles) {
    TF_ASSIGN_OR_RETURN(auto results, Execute(argument_handles));
    return std::pair<std::vector<std::unique_ptr<PyTpuBuffer>>, PyTpuToken>(
        std::move(results), PyTpuToken());
  }

  // Execute on local devices. Takes a sequence of argument lists (one argument
  // list per local device) and returns a tuple of results (one result per local
  // device). The number of argument lists must be equal to the local device
  // count.
  StatusOr<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>>
  ExecuteOnLocalDevices(
      absl::Span<const std::vector<PyTpuBuffer*>> argument_handles);

  StatusOr<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>>
  ExecuteShardedOnLocalDevices(
      absl::Span<const std::vector<PyTpuBuffer*>> args);

  StatusOr<std::pair<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>,
                     PyShardedTpuToken>>
  ExecuteShardedOnLocalDevicesWithTokens(
      absl::Span<const std::vector<PyTpuBuffer*>> args) {
    TF_ASSIGN_OR_RETURN(auto results, ExecuteShardedOnLocalDevices(args));

    TF_RET_CHECK(!args.empty());
    return std::pair<std::vector<std::vector<std::unique_ptr<PyTpuBuffer>>>,
                     PyShardedTpuToken>(std::move(results),
                                        PyShardedTpuToken());
  }

  void Delete() { executables_.clear(); }

 private:
  struct ExecuteResult {
    std::unique_ptr<PyTpuBuffer> buffer;
    std::shared_ptr<tpu_driver::Event> on_execute_finished;
  };

  ExecuteResult ExecuteHelper(
      absl::Span<const std::vector<PyTpuBuffer*>> all_core_arguments,
      absl::Span<PyTpuBuffer* const> this_core_arguments, int replica,
      int partition, const RunId& run_id);

  std::shared_ptr<PyTpuClient> const client_;
  std::map<int, std::unique_ptr<tpu_driver::LoadedProgramHandle>> executables_;
  const DeviceAssignment device_assignment_;
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
  std::vector<std::shared_ptr<PjRtDevice>> local_devices_;

  xla::Shape result_shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_CLIENT_TPU_CLIENT_H_
