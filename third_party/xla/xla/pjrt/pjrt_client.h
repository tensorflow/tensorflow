/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_CLIENT_H_
#define XLA_PJRT_PJRT_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

// API notes:
// PjRt stands for "Pretty much Just another RunTime".

namespace xla {

class PjRtClient;
class PjRtDevice;

class PjRtMemorySpace {
 public:
  virtual ~PjRtMemorySpace() = default;

  // The owner of this memory space.
  virtual PjRtClient* client() const = 0;

  // The devices that this memory space is attached to.
  virtual absl::Span<PjRtDevice* const> devices() const = 0;

  // The ID of this memory space. IDs are globally unique across all hosts.
  virtual int id() const = 0;

  // A platform-dependent string that uniquely identifies the kind of the
  // memory space.
  virtual absl::string_view kind() const = 0;

  // An ID uniquely identifies the kind of the memory space among those attached
  // to the same `PjRtClient`. The IDs assigned to a kind is implementation
  // specific.
  virtual int kind_id() const = 0;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current memory space unambiguously.
  virtual absl::string_view DebugString() const = 0;

  // Debug string suitable for reading by end users, should be reasonably terse.
  virtual absl::string_view ToString() const = 0;
};

class PjRtDevice {
 public:
  virtual ~PjRtDevice() = default;

  // Return the client that owns this device.
  virtual PjRtClient* client() const = 0;

  // Whether client can issue command to this device.
  virtual bool IsAddressable() const = 0;

  virtual const PjRtDeviceDescription& description() const {
    LOG(FATAL) << "PjRtDeviceDescription not available (must override "
                  "PjRtDevice::description).";
  }

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  ABSL_DEPRECATED("Use global_device_id() instead")
  virtual int id() const { return global_device_id().value(); }

  // There are several different IDs for a PJRT device.
  //
  // - global_device_id: The logical global device ID. This is unique among
  // devices of this type (e.g. CPUs, GPUs). On multi-host platforms, this will
  // be unique across all hosts' devices.  This is the ID that should be used in
  // a DeviceAssignment.
  //
  // - local_device_id: The logical local device ID. This will be used to look
  // up an addressable device local to a given client. It is -1 if undefined.
  //
  // - local_hardware_id: The physical local device ID, e.g., the CUDA device
  // number. Multiple PJRT devices can have the same local_hardware_id if
  // these PJRT devices share the same physical device. This is useful for
  // identifying which physical device when interacting with non-JAX code. In
  // general, not guaranteed to be dense, and -1 if undefined.

  // TODO(b/314368788): Remove `id()` and replace it with this function.
  virtual PjRtGlobalDeviceId global_device_id() const {
    return PjRtGlobalDeviceId(description().id());
  }

  virtual PjRtLocalDeviceId local_device_id() const {
    // By default, local_device_id is the same as local_hardware_id when there
    // is only one PJRT device on a physical device.
    return PjRtLocalDeviceId(local_hardware_id().value());
  }

  // Opaque hardware ID, e.g., the CUDA device number, useful for identifying
  // which GPU when interacting with non-JAX code. In general, not guaranteed to
  // be dense, and -1 if undefined.
  virtual PjRtLocalHardwareId local_hardware_id() const = 0;

  // The index of the process that this device belongs to, i.e. is addressable
  // from. This is not always identical to PjRtClient::process_index() in a
  // multi-process setting, where each client can see devices from all
  // processes, but only a subset of them are addressable and have the same
  // process_index as the client.
  virtual int process_index() const { return description().process_index(); }

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual absl::string_view device_kind() const {
    return description().device_kind();
  }

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  virtual absl::string_view DebugString() const {
    return description().DebugString();
  }

  // Debug string suitable for reading by end users, should be reasonably terse,
  // for example: "CpuDevice(id=0)".
  virtual absl::string_view ToString() const {
    return description().ToString();
  }

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the PjRtDevice.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const {
    return description().Attributes();
  }

  // Returns a scoped event that the caller uses to tell the PjRtClient that
  // there is asynchronous work happening that depends on activity on the
  // PjRtDevice. See comment on class definition in pjrt_future.h.
  //
  // Only some PjRtDevice implementations support ScopedAsyncTrackingEvent, and
  // those that do not will return nullptr.
  virtual std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const = 0;

  // Transfer the given literal to the infeed queue.
  virtual absl::Status TransferToInfeed(const LiteralSlice& literal) = 0;

  // Transfer and return a value of the given shape from the outfeed queue.
  virtual absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) = 0;

  // Returns allocator stats for the device. Only some PjRtDevice
  // implementations support allocator_stats, and those that do not will return
  // an Unimplemented error.
  virtual absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const {
    return Unimplemented("GetAllocatorStats is not supported");
  }

  // Returns all memory spaces attached to this device.
  // The memory spaces are in no particular order.
  virtual absl::Span<PjRtMemorySpace* const> memory_spaces() const = 0;

  // Returns the default memory space attached to this device.
  virtual absl::StatusOr<PjRtMemorySpace*> default_memory_space() const = 0;

  virtual absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view memory_space_kind) const {
    return Unimplemented("memory_space_by_kind not implemented");
  }

  // Returns a platform-specific stream handle that should be used to track when
  // an externally-managed buffer is ready to use on this device. This is
  // intended to support dlpack on GPU and is not expected to be implemented for
  // all hardware platforms.
  virtual absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const {
    return Unimplemented(
        "PjRtDevice::GetStreamForExternalReadyEvents only implemented for "
        "GPU");
  }

  // Experimental: Poisons the earliest execution on this device with given
  // launch_id if it's not finished yet, i.e. makes its output buffers error.
  //
  // Returns true if the output buffers have been successfully poisoned.
  //
  // Returns false if the output buffers were not successfully poisoned because
  // launch_id is not in the list of executions that have not yet completed.
  // This may happen either because the execution corresponding to launch_id has
  // already completed, or because an incorrect launch_id was supplied.
  //
  // Returns error otherwise, including in the case that poisoning is not
  // implemented by this client.
  virtual absl::StatusOr<bool> PoisonExecution(int32_t launch_id,
                                               absl::Status error) {
    return Unimplemented("PoisonExecution is not supported");
  }
};

// Forward declaration.
class PjRtBuffer;

// Helper struct for cross host transfers, returned by the callback from a call
// to PjRtBuffer::MakeCrossHostReceiveBuffers or
// PjRtBuffer::MakeCrossHostReceiveBuffersForGather.
struct PjRtCrossHostRecvDescriptors {
  // There is one serialized_descriptor per sub-buffer being gathered (i.e. a
  // single descriptor if the buffer is returned from a call to
  // MakeCrossHostReceiveBuffers). The descriptor should be transmitted to the
  // sender(s) and passed to a call to src_buffer->CopyToRemoteDevice.
  absl::InlinedVector<std::string, 1> serialized_descriptors;
};
// Function that the client should call at the receiver if it needs to cancel a
// cross-host send, for example because the buffer that the remote host wanted
// to send is not available. The serialized descriptor should match one of the
// descriptors returned in a PjRtCrossHostRecvDescriptors. on_canceled will be
// called once cancellation is complete and indicates whether cancellation was
// successful or not.
//
// For each serialized_descriptor provided in a PjRtCrossHostRecvDescriptors,
// *either* the sending host must successfully complete a CopyToRemoteDevice
// for that descriptor, *or* the receiving host must cancel. If there is a
// duplicate (e.g., both send and cancel) then the system will be left in an
// undefined state. If there is no send or cancellation then the system will
// hang indefinitely.
using PjRtCrossHostSendCancelNotifier = std::function<void(
    absl::string_view serialized_descriptor, absl::Status reason,
    std::function<void(absl::Status)> on_canceled)>;
// State asynchronously returned by MakeCrossHostReceiveBuffers. "descriptors"
// will match the returned PjRtBuffer objects 1:1. Specifically, each PjRtBuffer
// returned by MakeCrossHostReceiveBuffers will have one
// PjRtCrossHostRecvDescriptors object containing it descriptor(s).
struct PjRtCrossHostRecvState {
  std::vector<PjRtCrossHostRecvDescriptors> descriptors;
  PjRtCrossHostSendCancelNotifier cancel_notifier;
};
using PjRtCrossHostRecvNotifier =
    std::function<void(absl::StatusOr<PjRtCrossHostRecvState>)>;

// A sized chunk of host data. The host data can be either in host layout or in
// device layout, and it can be one part of the entire buffer. The PjRt
// implementations can customize how the memory is allocated and deallocated.
class PjRtChunk {
 public:
  // Allocate a PjRtChunk using malloc.
  static PjRtChunk AllocateDefault(size_t size) {
    return PjRtChunk(malloc(size), size, [](void* ptr) { free(ptr); });
  }

  PjRtChunk() = default;
  PjRtChunk(void* data, size_t size, std::function<void(void*)> deleter)
      : data_(static_cast<uint8_t*>(data)),
        size_(size),
        deleter_(std::move(deleter)) {}

  ~PjRtChunk() {
    if (data_) {
      deleter_(data_);
    }
  }

  PjRtChunk(PjRtChunk&& other)
      : data_(other.data_),
        size_(other.size_),
        deleter_(std::move(other.deleter_)) {
    other.data_ = nullptr;
  }
  PjRtChunk& operator=(PjRtChunk&& other) {
    if (data_) {
      deleter_(data_);
    }
    data_ = other.data_;
    size_ = other.size_;
    deleter_ = std::move(other.deleter_);
    other.data_ = nullptr;
    return *this;
  }

  PjRtChunk(const PjRtChunk&) = delete;
  PjRtChunk& operator=(const PjRtChunk&) = delete;

  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }
  int64_t size() const { return size_; }
  std::function<void(void*)> deleter() const { return deleter_; }

  // Release the ownership of the data. Note that this does not free the data;
  // the caller should copy `data()` and `deleter()` to manage the ownership
  // before calling `release()`. This PjRtChunk is invalidated after calling.
  void release() {
    data_ = nullptr;
    size_ = 0;
    deleter_ = nullptr;
  }

 private:
  // The ownership of the bytes pointed to by `data_` is controlled by the
  // `deleter_`.
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
  std::function<void(void*)> deleter_;
};

// A stream of Chunks from the host to the device. Once the stream enters
// Complete state it never changes state again.
//
// This class is thread-safe.
class CopyToDeviceStream {
 public:
  CopyToDeviceStream(int64_t total_bytes, int64_t granule_bytes)
      : total_bytes_(total_bytes), granule_bytes_(granule_bytes) {}

  virtual ~CopyToDeviceStream();

  // Emplaces a new Chunk of data to copy to the device. Returns an error future
  // if the Chunk's size causes the amount of transferred data to exceed
  // total_bytes(), if the stream is already complete, or if the chunk is not a
  // multiple of granule_size_in_bytes().
  //
  // The transfer is started immediately, and the returned future is fulfilled
  // when the transfer completes or fails.
  virtual PjRtFuture<> AddChunk(PjRtChunk chunk) = 0;

  // Returns the total amount of data the stream expects to be transferred.
  int64_t total_bytes() const { return total_bytes_; }

  // Returns the granule size in bytes. The size of the chunk added to this
  // stream must be a multiple of this number.
  int64_t granule_size_in_bytes() const { return granule_bytes_; }

  // Returns the amount of data the stream currently has either transferred or
  // has buffered to transfer.
  int64_t current_bytes() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return current_bytes_;
  }

  // Returns true if the stream is complete; all expected bytes have been
  // transferred or are buffered to transfer.
  bool IsComplete() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return IsCompleteLocked();
  }

  // Returns true if the stream is empty; no data has been queued.
  bool empty() const { return current_bytes() == 0; }

 protected:
  bool IsCompleteLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return current_bytes_ == total_bytes_;
  }

  int64_t total_bytes_;
  int64_t granule_bytes_;
  int64_t current_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  mutable absl::Mutex mu_;
};

class PjRtHostMemoryForDeviceManager {
 public:
  virtual ~PjRtHostMemoryForDeviceManager();

  // Transforms the host memory representations of a shape with the host layout
  // to the host memory representation of the same shape with the device layout.
  // `src_shape` and `dst_shape` may only differ in their layouts.
  virtual absl::StatusOr<PjRtChunk> ToDeviceLayout(
      const void* src_data, size_t src_size, const Shape& host_shape,
      const Shape& device_shape) = 0;

  // Transforms the host memory representations of a shape with the device
  // layout to the host memory representation of the same shape with the host
  // layout. `src_shape` and `dst_shape` may only differ in their layouts.
  virtual absl::Status ToHostLayout(const void* src_data, size_t src_size,
                                    const Shape& src_shape, void* dst_data,
                                    size_t dst_size,
                                    const Shape& dst_shape) = 0;
};

class PjRtLoadedExecutable;

struct PjRtPluginAttributes {
  int64_t pjrt_c_api_major_version;
  int64_t pjrt_c_api_minor_version;
  absl::flat_hash_map<std::string, PjRtValueType> attributes;
};

// Encapsulates the state of Python session with XLA.
//
// It is the responsibility of the client of this API to keep the PjRtClient
// alive as long as any of the other runtime objects are alive.
//
// A note on the semantics of cross-device copies.
//
// There are two mechanisms to transfer a buffer from one device to another.
// When both devices are on the same host (more specifically, the user program
// ends up with pointers to both the source and destination buffers in the same
// address space), the caller can use:
//   dst_buffer = src_buffer->CopyToDevice(dst_device)
//
// When the source and destination are on different hosts, but the transfer is
// made via native device networking (as opposed to the user program fetching
// the buffer and sending it using its own networking code), the caller can
// use:
//   DstHost: dst_client->MakeCrossHostReceiveBuffers(...)
//   DstHost: [...]
//   DstHost: gets callback containing PjRtCrossHostRecvDescriptors
//   DstHost: sends cross-host recv serialized descriptors to SrcHost
//   SrcHost: src_buffer->CopyToRemoteDevice(serialized_descriptors)
//
// Note that in the cross-host case, the dst_client may call
// MakeCrossHostReceiveBuffers before the action that produces src_buffer has
// been enqueued at SrcHost.
//
// On some platforms, device-to-device transfers consume scarce hardware
// resources. If dst_client->MakeCrossHostReceiveBuffers immediately claimed
// those resources, then there would be a risk of system-wide deadlock, if the
// resources claimed by the recv prevented other transfers that are necessary
// to generate src_buffer from acquiring enough resources to proceed.
//
// In order to allow clients to avoid deadlocks such as those in the preceding
// paragraph, PjRtClient guarantees progress but not fairness with respect to
// the order that cross-device transfers are enqueued on a given host, as
// follows:
//
// The progress guarantee is that a cross-device transfer T on host A will not
// claim scarce hardware resources until it is guaranteed that all transfers
// enqueued on A before T have already either completed, or been assigned enough
// resources to ensure that they can eventually complete.
//
// The lack of a fairness guarantee means that, if cross-device transfer T1 is
// enqueued before transfer T2 at A, then T2 may complete before T1. T1 may be
// delayed for an unbounded time waiting for T2 if T2 is large, even though T1
// will eventually be able to make progress.
class PjRtClient {
 public:
  struct ShapeSpec {
    PrimitiveType element_type;
    DimensionVector dims;
  };

  PjRtClient() = default;
  explicit PjRtClient(std::unique_ptr<PjRtHostMemoryForDeviceManager>
                          host_memory_for_device_manager)
      : host_memory_for_device_manager_(
            std::move(host_memory_for_device_manager)) {}

  virtual ~PjRtClient() = default;

  // Return the process index of this client. Always 0 in single-process
  // settings.
  virtual int process_index() const = 0;

  // Return the number of devices in the entire computation. In multi-headed
  // client setting, some are addressable by this client, some are not. In a
  // single-client setting, this is equal to the number of addressable devices.
  virtual int device_count() const = 0;

  // Return number of addressable devices. Addressable devices are those that
  // the client can issue commands to.
  virtual int addressable_device_count() const = 0;

  // Return all devices known to the client, including addressable and
  // non-addressable devices.
  virtual absl::Span<PjRtDevice* const> devices() const = 0;

  // Return only addressable devices. The devices are in no particular order.
  virtual absl::Span<PjRtDevice* const> addressable_devices() const = 0;

  // Lookup any PjRtDevice for a given PjRtDevice::id().
  virtual absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const {
    return Unimplemented("LookupDevice is not supported.");
  }

  // Return an addressable PjRtDevice for a given
  // PjRtDevice::local_device_id().
  virtual absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const {
    return Unimplemented("LookupAddressableDevice is not supported.");
  }

  // Return all memory spaces owned by the client.
  // The memory spaces are in no particular order.
  virtual absl::Span<PjRtMemorySpace* const> memory_spaces() const = 0;

  // Return an ID that identifies the platform (CPU/GPU/TPU).
  virtual PjRtPlatformId platform_id() const = 0;

  // Returns a string that identifies the platform (CPU/GPU/TPU).
  virtual absl::string_view platform_name() const = 0;

  // Returns a string containing human-readable, platform-specific version info
  // (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  virtual absl::string_view platform_version() const = 0;

  // Returns the key value store used by the client.
  virtual std::optional<std::shared_ptr<KeyValueStoreInterface>>
  key_value_store() const {
    return std::nullopt;
  }

  // Returns information about the underlying PJRT C API plugin if such a plugin
  // is being used, otherwise returns nullopt.
  virtual std::optional<PjRtPluginAttributes> plugin_attributes() const {
    return std::nullopt;
  }

  // Return a device-specific default device assignment, e.g., GPU and TPU may
  // be different.
  virtual absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const {
    return Unimplemented("GetDefaultDeviceAssignment is not supported.");
  }

  // Returns a device-specific default device assignment for multi-slice system.
  // If num_replicas_per_slice is not defined (nullopt) then we assume that
  // all the partitions live entirely on a single slice and that all cross slice
  // communication happens across replicas assuming then that
  // num_replicas_per_slice is going to be "num_replicas / num_slices".
  // TODO(zhangqiaorjc): Convert this to pure virtual and push down.
  virtual absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, std::optional<int> num_replicas_per_slice,
      int num_partitions, const MultiSliceConfig* multi_slice_config) const {
    return Unimplemented("Multi slice device assignment is not supported.");
  }

  // Returns the default device layout for a buffer with `element_type` and
  // `dims`. The default layout is a platform-specific layout used when no other
  // layout is specified, e.g. for host-to-device transfers. When compiling, the
  // default layout is used for program arguments and outputs unless
  // user-specified or compiler-chosen layouts are requested via the
  // "mhlo.layout_mode" attribute.
  virtual absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) {
    return Unimplemented("GetDefaultLayout is not supported.");
  }

  // Returns a backend-specific HLO cost analysis visitor.
  virtual absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const {
    return Unimplemented("GetHloCostAnalysis is not supported.");
  }

  // Compile `computation` with given `options`.
  virtual absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) {
    return Unimplemented("Compile with options is not supported.");
  }

  // Variant of `Compile` that accepts an MLIR module.
  virtual absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) {
    return Unimplemented("Compile with MLIR Module is not supported.");
  }

  // Deserializes a serialized executable as produced by
  // PjRtExecutable::SerializeExecutable(). `serialized` must have been
  // produced by a compiler of the same platform and version as this one.
  //
  // Pending completion of b/237720161, `options` is a mandatory argument in
  // most implementations of this interface. They _are_ optional for
  // implementations related to the PJRT C API.
  virtual absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  DeserializeExecutable(absl::string_view serialized,
                        std::optional<CompileOptions> options) {
    return Unimplemented("Deserialize is not supported.");
  }

  // LoadSerializedExecutable takes the serialized output of PjRtExecutable. The
  // returned executable is loaded by this client. The same checks are made as
  // in Load that the serialized executable is compatible with the client.
  virtual absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) {
    return Unimplemented("Loading serialized executable not supported.");
  }

  // Loads the executable returns aa PjRtLoadedExecutable runnable by this
  // client. Returns an error if the PjRtExecutable was created with an
  // incompatible topology or client.
  // PjRtExecutable contains a copy of the CompileOptions that was used to
  // generate the executable. Load will use the CompileOptions from within the
  // executable.
  virtual absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) {
    return Unimplemented("Loading executable not supported.");
  }

  // Creates a buffer on the device without initializing or copying any data.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) {
    return Unimplemented("CreateUnitializedBuffer is not supported.");
  }

  // Creates buffer in the given memory space that carries an error future
  // without allocating memory.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtMemorySpace* memory) {
    return Unimplemented("CreateErrorBuffer not supported.");
  }

  // Creates buffer in the given device that carries an error future without
  // allocating memory.
  ABSL_DEPRECATED(
      "Use CreateErrorBuffer(absl::Status, Shape, PjRtMemorySpace*)")
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateErrorBuffer(
      absl::Status error, const Shape& shape, PjRtDevice* device) {
    auto default_memory_space = device->default_memory_space();
    if (!default_memory_space.ok()) {
      return default_memory_space.status();
    }
    return CreateErrorBuffer(std::move(error), shape, *default_memory_space);
  }

  // Gets the pointer to the topology description held by the client.
  virtual absl::StatusOr<const PjRtTopologyDescription*>
  GetTopologyDescription() const {
    return Unimplemented("GetTopologyDescription not supported on platform %s",
                         platform_name());
  }

  // A client may want to create a buffer, and hand the buffer to other PjRt
  // methods, before the data to store in the buffer is available to the client.
  // This is supported using CreateBuffersForAsyncHostToDevice, which returns an
  // AsyncHostToDeviceTransferManager helper object.
  //
  // The PjRtBuffers can be retrieved from the AsyncHostToDeviceTransferManager
  // and safely passed immediately to downstream PjRt method calls. Subsequently
  // the client can call methods on the AsyncHostToDeviceTransferManager object
  // to copy data into the buffers, and once the data copies are complete, the
  // buffers' definition events will automatically become ready, unblocking
  // downstream consumers of the buffers.
  //
  // Depending on the backend's implementation, a single call to
  // CreateBuffersForAsyncHostToDevice may either:
  //   - Create a "batch" of buffers that share a single definition event, which
  //   may amortize some performance overheads, but means that none of the
  //   buffers are available to downstream consumers until all the transfers
  //   have completed, in which case multiple calls to
  //   CreateBuffersForAsyncHostToDevice should be made if it is desirable for
  //   buffers to become available as soon as transfers into them complete.
  //
  //   - Create a "batch" of buffers with multiple underlying definitions
  //   events, and individual buffers become available to downstream consumers
  //   as soon as transfers into them complete.

  // Helper class to all clients to asynchronously transfer data into buffers
  // that are created uninitialized, see comments immediately above.
  class AsyncHostToDeviceTransferManager {
   public:
    virtual ~AsyncHostToDeviceTransferManager() = default;

    // Returns the number of buffers managed by this object.
    virtual size_t buffer_count() const = 0;

    // Returns the destination device of the transfers.
    virtual PjRtDevice* device() const = 0;

    // Returns buffer_index, which can be passed to downstream consumers
    // immediately and will become available once transfers complete. May not
    // be called more than once for a given buffer_index.
    //
    // RetrieveBuffer can be called at any convenient time; transfer methods
    // can safely be called for a buffer index after RetrieveBuffer has been
    // called.
    virtual std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) = 0;

    // Transfers 'literal' into buffer_index. No transfer calls into
    // buffer_index can be made after this call. on_done is called when the
    // transfer is complete but before the buffers are made available to
    // their consumers. 'literal' must remain in scope until on_done is
    // called.
    virtual absl::Status TransferLiteralToBuffer(
        int buffer_index, const LiteralSlice& literal,
        absl::AnyInvocable<void() &&> on_done) = 0;

    // Returns the on-device size in bytes of buffer buffer_index.
    virtual size_t buffer_size(int buffer_index) const = 0;

    // Transfers 'data' into buffer_index. 'data' must be already laid out in
    // the correct on-device format, for example returned by a call to
    // buffer->CopyRawToHost. No transfer calls (or SetBufferError calls) into
    // buffer_index can be made after this call. on_done is called when the
    // transfer is complete but before the buffers are made available to their
    // consumers. 'data' must remain in scope until on_done is called.
    virtual absl::Status TransferRawDataToBuffer(
        int buffer_index, absl::string_view data,
        absl::AnyInvocable<void() &&> on_done) = 0;

    // Transfers 'data' into a sub-buffer of buffer_index starting at offset, of
    // length transfer_size. 'data' must be already laid out in the correct
    // on-device format, for example returned by a call to
    // buffer->CopyRawToHost. If is_last_transfer is false then the buffer
    // remains unavailable to consumers after the transfer completes. If
    // is_last_transfer is true then the buffer becomes available to consumers
    // after the transfer completes, and no transfer calls (or SetBufferError
    // calls) into buffer_index can be made after this call. on_done is called
    // when the transfer is complete but before the buffers are made available
    // to their consumers. 'data' must remain in scope until on_done is called.
    virtual absl::Status TransferRawDataToSubBuffer(
        int buffer_index, const void* data, int64_t offset,
        int64_t transfer_size, bool is_last_transfer,
        absl::AnyInvocable<void() &&> on_done) = 0;

    // Indicates that a specific buffer should result in an error status. No
    // transfer calls (or further SetBufferError calls) into buffer_index can
    // be made after this call.
    virtual void SetBufferError(int buffer_index, absl::Status error) = 0;

    // Adds the specified key/value metadata for the transfer operation.
    // This is typically used for debugging purposes, such as adding a handle
    // that can be used to identify transfer operations.
    using TransferMetadata = absl::flat_hash_map<std::string, std::string>;
    virtual void AddTransferMetadata(const TransferMetadata& metadata) = 0;
  };

  // Returns a manager for async transfers into a set of buffers with on-host
  // shapes defined by 'shape_specs' and optional `device_layouts`.
  //
  // If the desired layout of one or more buffers is not specified in
  // `device_layouts`, then those buffers will use the default device layout. If
  // `device_layouts` itself is not specified, then all buffers will use the
  // default device layout.
  virtual absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtDevice* device) {
    return absl::UnimplementedError(absl::StrCat(
        "CreateBuffersForAsyncHostToDevice with ShapeSpec and Layout is "
        "not implemented on platform: ",
        platform_name()));
  }

  // Variant of CreateBuffersForAsyncHostToDevice with PjRtMemorySpace.
  virtual absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) {
    return absl::UnimplementedError(absl::StrCat(
        "CreateBuffersForAsyncHostToDevice with ShapeSpec and Layout is "
        "not implemented on platform: ",
        platform_name()));
  }

  // Returns a manager for async transfers into a set of buffers with on-host
  // shapes 'shapes'.
  virtual absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) {
    return Unimplemented(
        "CreateBuffersForAsyncHostToDevice with on host is not implemented.");
  }

  // Variant of CreateBuffersForAsyncHostToDevice with PjRtMemorySpace.
  virtual absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) {
    return Unimplemented(
        "CreateBuffersForAsyncHostToDevice with PjRtMemorySpace is not "
        "implemented.");
  }

  // Creates a shapeless buffer on the device that can be partitioned into
  // multiple PjRtBuffer. This class is an Arena version of
  // `AsyncHostToDeviceTransferManager`.
  // As a low-level interface, the user must make sure that invocations of
  // `Slice` match properly with the writes from `TransferRawDataToSubBuffer`.
  //
  // For the intended application to Arena allocation / transfer, the user can
  // use `GetOnDeviceSizeInBytes` to calculate the offsets for the host buffers
  // that need to be transferred.
  class PjRtRawDeviceBuffer {
   public:
    virtual ~PjRtRawDeviceBuffer() = default;

    // Transfers data to the device buffer. Data should already be in the
    // device layout.
    virtual absl::Status TransferRawDataToSubBuffer(
        const void* data, int64_t offset, int64_t transfer_size,
        bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) = 0;

    // The resulting buffer becomes ready when all transfers complete.
    virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> Slice(
        int64_t offset, PrimitiveType type, absl::Span<int64_t const> dims,
        const Layout& layout) = 0;
  };
  // Creates a raw device buffer of a given size in bytes.
  virtual absl::StatusOr<std::unique_ptr<PjRtRawDeviceBuffer>>
  CreateRawDeviceBuffer(int64_t size, PjRtDevice* device) {
    return Unimplemented("CreateRawDeviceBuffer is not implemented.");
  }

  // On-device bytes required for a PjRt buffer with these `Shape` attributes.
  virtual absl::StatusOr<int64_t> GetOnDeviceSizeInBytes(
      PrimitiveType type, absl::Span<int64_t const> dims,
      const Layout& layout) {
    return Unimplemented("GetOnDeviceSizeInBytes is not implemented.");
  };

  // Describes the semantics the caller to BufferFromHostBuffer expects from the
  // runtime, in a total order from most restrictive to least restrictive.
  enum class HostBufferSemantics {
    // The runtime may not hold references to `data` after the call to
    // `BufferFromHostBuffer` completes. The caller promises that `data` is
    // immutable and will not be freed only for the duration of the
    // BufferFromHostBuffer call. `on_done_with_host_buffer` will be called
    // before `BufferFromHostBuffer` returns.
    kImmutableOnlyDuringCall,

    // The runtime may hold onto `data` after the call to `BufferFromHostBuffer`
    // returns while the runtime completes a transfer to the device. The caller
    // promises not to mutate or free `data` until the transfer completes, at
    // which point the runtime will call `on_done_with_host_buffer`. It is also
    // correct to wait on the host (directly or indirectly) for the buffer's
    // definition event to complete.
    kImmutableUntilTransferCompletes,

    // The PjRtBuffer may alias `data` internally and the runtime may use the
    // `data` contents as long as the buffer is alive. The runtime promises not
    // to mutate contents of the buffer (i.e. it will not use it for aliased
    // output buffers). The caller promises to keep `data` alive and also not to
    // mutate its contents as long as the buffer is alive; to notify the caller
    // that the buffer may be freed, the runtime will call
    // `on_done_with_host_buffer` when the PjRtBuffer is freed. On non-CPU
    // platforms this acts identically to kImmutableUntilTransferCompletes.
    kImmutableZeroCopy,

    // The PjRtBuffer may alias `data` internally and the runtime may use the
    // `data` contents as long as the buffer is alive. The runtime is allowed
    // to mutate contents of the buffer (i.e. use it for aliased output
    // buffers). The caller promises to keep `data` alive and not to mutate its
    // contents as long as the buffer is alive (otherwise it could be a data
    // race with the runtime); to notify the caller that the buffer may be
    // freed, the runtime will call `on_done_with_host_buffer` when the
    // PjRtBuffer is freed. On non-CPU platforms this acts identically to
    // kImmutableUntilTransferCompletes.
    kMutableZeroCopy,
  };

  // on_done_with_host_buffer is optional and may be null.
  // on_done_with_host_buffer will be called iff an OK status is returned.
  //
  // `data` points to the backing array of the host buffer. Caution:
  // `byte_strides` are allowed to be negative, in which case `data` may need
  // to point to the interior of the buffer, not necessarily its start.
  //
  // If byte_strides is omitted, the array is assumed to have a dense layout
  // with dimensions in major-to-minor order.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device) {
    return Unimplemented("BufferFromHostBuffer is not implemented.");
  }

  // Variant of BufferFromHostBuffer that takes an optional device layout. It is
  // used when non-compact layout is preferred.
  // TODO(b/275645543): remove BufferFromHostBuffer without optional device
  // layout after all the inherited classes and call sites are updated.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device, const Layout* device_layout) {
    return tsl::errors::Unimplemented(
        "BufferFromHostBuffer with an optional device layout is not "
        "implemented on platform: ",
        platform_name());
  }

  // TODO(b/277820585): remove BufferFromHostBuffer with PjRtDevice after the
  // migration is done.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) {
    return tsl::errors::Unimplemented(
        "BufferFromHostBuffer with PjRtMemorySpace is not implemented on "
        "platform: ",
        platform_name());
  }

  // Note that literal must remain in scope until the transfer has completed, so
  // the caller should, for example, wait for GetReadyFuture().Await()
  // completes on the return value before letting literal go out of scope.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) {
    return Unimplemented("BufferFromHostLiteral is not implemented.");
  }

  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device,
      const Layout* device_layout) {
    if (device_layout) {
      return absl::UnimplementedError(absl::StrCat(
          "BufferFromHostLiteral with device_layout is not implemented on "
          "platform: ",
          platform_name()));
    }

    return this->BufferFromHostLiteral(literal, device);
  }

  // TODO(b/277820585): remove BufferFromHostLiteral with PjRtDevice after the
  // migration is done.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space) {
    return tsl::errors::Unimplemented(
        "BufferFromHostLiteral with PjRtMemorySpace is not implemented on "
        "platform: ",
        platform_name());
  }

  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) {
    if (device_layout) {
      return absl::UnimplementedError(absl::StrCat(
          "BufferFromHostLiteral with device_layout is not implemented on "
          "platform: ",
          platform_name()));
    }
    return this->BufferFromHostLiteral(literal, memory_space);
  }

  // Creates a PjRtBuffer that is a non-owned view of an on-device
  // buffer (typically allocated by another library).
  // on_delete_callback is called when the PjRtBuffer is done with the on-device
  // buffer. The buffer may be mutated, for example, if the buffer is donated
  // to an Execute operation.
  //
  // `stream`, if specified, is a platform-specific stream handle that should
  // contain the work or events needed to materialize the on-device
  // buffer. CreateViewOfDeviceBuffer will append an event to `stream` that
  // indicates when the returned buffer is ready to use. This is intended to
  // support dlpack on GPU and is not expected to be supported on all hardware
  // platforms.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream = std::nullopt) {
    return Unimplemented("CreateViewOfDeviceBuffer is not implemented.");
  }

  // Returns platform-dependent address for the given buffer that is often but
  // not guaranteed to be the physical/device address.
  virtual absl::StatusOr<std::uintptr_t> UnsafeBufferPointer(
      PjRtBuffer* buffer);

  // Returns a vector of PjRtBuffers that can be used to receive
  // cross host transfers using `client` on `device'. Asynchronously calls
  // `notifier` once receive descriptors are ready to be communicated to the
  // sender. `shapes` must be the exact shapes, with identical layouts,
  // corresponding to the buffers that will be sent. When resources for the
  // transfer are available, notifier will be called with a vector of
  // PjRtCrossHostRecvDescriptors structs, one for each shape in `shapes`. Each
  // struct contains an opaque string that should be transmitted to the sending
  // host and used in a call to CopyToRemoteDevice. None of the recv buffers
  // will become ready until *all* of the sends have completed.
  //
  // If MakeCrossHostReceiveBuffers returns an error, then `notifier` will not
  // be called. Otherwise `notifier` will be called exactly once. In the case
  // where `notifier` is called with an error status, then the PjRtBuffers
  // returned by MakeCrossHostReceiveBuffers will never yield data.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  virtual absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) {
    return Unimplemented("MakeCrossHostReceiveBuffers is not implemented.");
  }

  // Asynchronously makes a vector of PjRtBuffers that can be used to receive
  // cross host transfers, as in MakeCrossHostReceiveBuffers above, however
  // each buffer expects to be "gathered" using multiple sends, one for each of
  // a set of sub-slices of the destination buffer.
  //
  // For each value in shapes there is a corresponding FullGatherDetails struct
  // that describes the sub-slices.
  struct GatherDetails {
    // The dimensions of the corresponding buffer that the gather slices
    // into. These dimensions must be the major dimensions in the on-device
    // layout of the buffer, and must all be untiled. The scatter acts as if
    // the buffer were transposed/reshaped so that all of these dimensions were
    // combined into a single dimension whose size is the product of the
    // dimensions, and the slice indices correspond to indices in that single
    // combined dimension.
    //
    // For example, if the shape is [3, 4, 128, 128] with [3, 4] as the major
    // dimensions in the layout, and dimensions = {0, 1}, then the buffer is
    // treated as if it were shape [12, 128, 128] and the indices in
    // slice_boundaries range in [0, 12].
    absl::InlinedVector<int, 3> dimensions;
    // The cumulative indices in dimension of the slices. For example, if
    // shape.dimensions(dimension)==10, setting slice_boundaries to {2, 5, 10}
    // would correspond to 3 slices of sizes {2, 3, 5} respectively. If the last
    // entry in slice_boundaries is less than the size of the combined gather
    // dimension, the trailing data in the buffer is undefined after the receive
    // completes.
    std::vector<int64_t> slice_boundaries;
  };
  virtual absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) {
    return Unimplemented(
        "MakeCrossHostReceiveBuffersForGather is not implemented.");
  }

  // Create ChannelHandles for XLA send/recv.
  virtual absl::StatusOr<ChannelHandle> CreateChannelHandle() {
    return Unimplemented("CreateChannelHandle is not implemented.");
  }
  virtual absl::StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() {
    return Unimplemented("CreateDeviceToHostChannelHandle is not implemented.");
  }

  // TODO(zhangqiaorjc): Experimental API to be removed.
  // Defragment device memory.
  virtual absl::Status Defragment() {
    return Unimplemented("Defragment is not implemented.");
  }

  // If false, this client does not support send/recv host callbacks, and
  // callers should not set the `send_callbacks` and `recv_callbacks` arguments
  // in ExecuteOptions.
  virtual bool SupportsSendRecvCallbacks() const { return false; }

  // Return the PjRtHostMemoryForDeviceManager for this client. It can be
  // nullptr if the implementation does not provide one.
  virtual PjRtHostMemoryForDeviceManager* GetPjRtHostMemoryForDeviceManager()
      const {
    return host_memory_for_device_manager_.get();
  }

 private:
  std::unique_ptr<PjRtHostMemoryForDeviceManager>
      host_memory_for_device_manager_;
};

// Holds a reference from Python to a tuple of device buffers. A PjRtBuffer
// can be either valid or invalid. An invalid buffer is one that has never been
// initialized, or a buffer that has been deleted (e.g., by calling Delete, or
// by donating it to a computation that aliases an input parameter to an
// output). We allow PjRtBuffer objects to outlive the underlying device
// buffers so we can decouple buffer lifetimes from the corresponding Python
// references if needed. Thread-safe.
class PjRtBuffer {
 public:
  virtual ~PjRtBuffer() = default;

  virtual PrimitiveType element_type() const {
    return on_device_shape().element_type();
  }

  // Returned dimensions have lifetime of this buffer.
  virtual absl::Span<const int64_t> dimensions() const {
    return on_device_shape().dimensions();
  }

  // The on-device memory layout of this buffer. Returned via unique_ptr to make
  // memory management easier -- PjRtLayout is an abstract base class, so cannot
  // be easily copied.
  virtual std::unique_ptr<PjRtLayout> layout() const {
    CHECK(on_device_shape().has_layout());
    return std::make_unique<PjRtXlaLayout>(on_device_shape().layout());
  }

  // PjRtBuffers can either represent a single array buffer or a tuple of array
  // buffers. Returns true if this buffer represents a tuple, false if an array.
  virtual bool IsTuple() const { return on_device_shape().IsTuple(); }

  virtual const Shape& on_device_shape() const = 0;

  virtual bool has_dynamic_dimensions() const {
    return on_device_shape().is_dynamic();
  }

  // Each returned element is true if the corresponding dimensions is dynamic,
  // false if static.
  virtual absl::Span<const bool> is_dynamic_dimension() const {
    return on_device_shape().dynamic_dimensions();
  }

  // Same as dimensions() when the shape is static. When the shape is dynamic,
  // it gathers the metadata from the device and returns a static shape
  // representing the logical shape of the data. This approach is identical to
  // how tensorflow and xrt setup the output buffer in the graph.
  //
  // Since this method actually acquires locks and communicate with the device,
  // it does not have the const qualifier, similar to what ToLiteral does.
  virtual absl::StatusOr<std::vector<int64_t>> logical_dimensions() {
    TF_ASSIGN_OR_RETURN(Shape logical_shape, logical_on_device_shape());
    absl::Span<const int64_t> dims = logical_shape.dimensions();
    return std::vector<int64_t>(dims.begin(), dims.end());
  }

  // Same as on_device_shape when the shape is static. When the shape is
  // dynamic, it gathers the metadata from the device and returns a static shape
  // representing the logical shape of the data. This approach is identical to
  // how tensorflow and xrt setup the output buffer in the graph.
  //
  // Since this method actually acquires locks and communicate with the device,
  // it does not have the const qualifier, similar to what ToLiteral does.
  virtual absl::StatusOr<Shape> logical_on_device_shape() {
    const Shape& shape = on_device_shape();
    CHECK(shape.is_static())
        << "logical_on_device_shape needs to be overridden for platform '"
        << client()->platform_name() << "'";
    return shape;
  }

  virtual PjRtMemorySpace* memory_space() const = 0;
  // TODO(b/277820585): remove device() after the migration is done.
  virtual PjRtDevice* device() const = 0;
  virtual PjRtClient* client() const = 0;

  // ExternalReference is a potentially long-lived reference held while a buffer
  // is being shared by an external framework, e.g., NumPy. A client acquires an
  // external reference by calling PjRtBuffer::AcquireExternalReference() and
  // releases it by deleting the ExternalReference. The external framework
  // should not modify the underlying buffer unless it is confident via its own
  // synchronization that modifications do not race with reads from the
  // PjRtBuffer.
  class ExternalReference {
   public:
    virtual ~ExternalReference() = 0;
    // Return opaque device memory pointer to root buffer.
    void* OpaqueDeviceMemoryDataPointer() const { return data_ptr_; }

    // Stream is platform-specific. This is intended to support dlpack on GPU
    // and is not expected to be implemented for all hardware platforms.
    virtual absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) {
      return Unimplemented(
          "WaitUntilBufferReadyOnStream is only implemented for GPU.");
    }

   protected:
    void* data_ptr_;
  };
  virtual absl::StatusOr<std::unique_ptr<ExternalReference>>
  AcquireExternalReference() = 0;

  // Asynchronously copies the buffer's value into `literal`.
  //
  // Return value is a future the caller can use to discover when the copy has
  // completed. The transfer respects the layout of `literal`; to specify a
  // particular layout, set the layout before calling `ToLiteral`.
  virtual PjRtFuture<> ToLiteral(MutableLiteralBase* literal) = 0;
  // This version of ToLiteral allows the implementation to defer the
  // construction of the literal (e.g. until the underlying buffer is ready).
  // The specific timing of calling `generator` is implementation defined, and
  // might be done eagerly, but it is guaranteed to be earlier than when the
  // returned future becomes ready.
  virtual PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&>
          generator) = 0;

  // Synchronous overload of ToLiteral, as a convenience.
  absl::Status ToLiteralSync(MutableLiteralBase* literal) {
    absl::Notification done;
    absl::Status status;
    ToLiteral(literal).OnReady([&](absl::Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    return status;
  }

  absl::StatusOr<Shape> HostShape() {
    Shape device_shape;
    if (!IsTuple()) {
      absl::Span<const int64_t> literal_dims;
      std::optional<std::vector<int64_t>> logical_dims_storage;
      if (has_dynamic_dimensions()) {
        TF_ASSIGN_OR_RETURN(std::vector<int64_t> logical_dims,
                            logical_dimensions());
        logical_dims_storage.emplace(std::move(logical_dims));
        literal_dims = *logical_dims_storage;
      } else {
        literal_dims = dimensions();
      }
      if (element_type() == TOKEN) {
        device_shape = ShapeUtil::MakeTokenShape();
      } else {
        device_shape = ShapeUtil::MakeShape(element_type(), literal_dims);
        // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
        *device_shape.mutable_layout() = GetXlaLayoutUnsafe(layout());
      }
    } else {
      // TODO(skyewm): does anything need to create tuple literals? The PJRT C
      // API doesn't support tuples or {logical_}on_device_shape(), so we prefer
      // to use the above non-tuple code path where possible.
      device_shape = on_device_shape();
      if (device_shape.is_dynamic()) {
        TF_ASSIGN_OR_RETURN(device_shape, logical_on_device_shape());
      }
    }
    return ShapeUtil::DeviceShapeToHostShape(device_shape);
  }

  // Convenience synchronous overload that allocates a literal with a default
  // layout.
  absl::StatusOr<std::shared_ptr<Literal>> ToLiteralSync() {
    TF_ASSIGN_OR_RETURN(Shape host_shape, HostShape());
    auto literal = std::make_shared<Literal>(host_shape);
    TF_RETURN_IF_ERROR(ToLiteralSync(literal.get()));
    return literal;
  }

  // Returns the number of bytes of the buffer storage on the device.
  virtual absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const = 0;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `dst` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  virtual PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                                     int64_t transfer_size) = 0;

  // As above, but the transfer will not happen until `dst` is fulfilled with a
  // valid pointer. If `dst` is fulfilled with a non-Ok status, then the
  // transfer will be cancelled. The implementation must ensure that the
  // underlying buffer is kept alive even if the `PjRtBuffer` is deleted before
  // the `dst` future is fulfilled.
  //
  // In error cases it is possible for the returned Future to become ready
  // before `dst` is fulfilled.
  //
  // The default implementation always returns a future that is fulfilled with
  // an UNIMPLEMENTED error.
  virtual PjRtFuture<> CopyRawToHostFuture(PjRtFuture<void*> dst,
                                           int64_t offset,
                                           int64_t transfer_size);

  // Drops the buffer's reference to its associated device memory, leaving the
  // buffer in an invalid state. The memory will be freed lazily when all async
  // operations using the buffer have completed, according to the allocation
  // semantics of the underlying platform. Delete may briefly block if another
  // thread is in the process of enqueuing an operation on this buffer, but it
  // will never block for a stream operation to complete. If an external
  // framework holds a reference to the TrackedDeviceBuffer via
  // GetBufferWithExternalReference, the memory will not be freed until the
  // external framework drops the reference.
  virtual void Delete() = 0;

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but transfers the device
  // memory ownership out via an ExternalReference rather than
  // freeing the device memory, so that another framework can take ownership of
  // it. A return value of nullptr indicates that PjRtBuffer has been
  // deleted. The buffer returned from Release may be safely dropped at any time
  // even if it still has pending async operations. The client should call
  // GetReadyFuture().Await before calling ReleaseDeviceMemoryOwnership with
  // wait_for_operations_to_complete=false, to ensure that the host has
  // synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from ReleaseDeviceMemoryOwnership.
  virtual absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) = 0;

  // True if and only if Delete or Release has previously been called.
  virtual bool IsDeleted() = 0;

  // Copies the buffer to device `dst_device`, performing a d2d transfer when
  // `dst_device` is sharing the same Client, and performing a d2h and h2d copy
  // if `dst_device` lives on a different Client.
  // Returns an error if the buffer is already on dst_device.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) = 0;

  // Copies the buffer to memory space `dst_memory_space`.
  //
  // The destination memory space may be attached to any client, but optimized
  // implementations may apply when the copy is within the same client.
  //
  // Returns an error if the buffer is already in dst_memory_space.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) = 0;

  // Prepares to send a copy of the buffer to a remote device. The destination
  // device is encoded in `serialized_descriptor`, which must be fulfilled by
  // the result of call to MakeCrossHostReceiveBuffers on the remote host's
  // destination device. MakeCrossHostReceiveBuffers takes an array of shapes to
  // construct the destination buffers, and a callback supplies an array
  // containing both the destination buffers, and a serialized descriptor for
  // each buffer. For each destination buffer there should be a matching call to
  // src->CopyToRemoteDevice on a remote host for a src buffer of the
  // corresponding shape. If `serialized_descriptor` is fulfilled with a non-Ok
  // status, then the transfer is canceled, otherwise it must be the string
  // returned by the MakeCrossHostReceiveBuffers callback corresponding to the
  // destination buffer.
  //
  // When the send either completes or fails, `on_done` will be called. If
  // `status` is Ok then it is guaranteed that sends_were_enqueued==true.
  // Otherwise, if sends_were_enqueued==false then the sender should contact
  // the receiver out of band to request cancellation of the transfer. If
  // !status.ok() and sends_were_enqueued==true then it is not possible to
  // determine whether the transfer succeeded and the system is in an
  // undefined state. This undefined state almost certainly indicates an
  // unrecoverable hardware error. Note that in some error cases, `on_done` may
  // be called before `serialized_descriptor` is fulfilled.
  //
  // Some implementations of this method may immediately block on the
  // `serialized_descriptor` future (and not return until that future has been
  // fulfilled).
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  using RemoteSendCallback =
      std::function<void(absl::Status status, bool sends_were_enqueued)>;
  virtual void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                                  RemoteSendCallback on_done) = 0;
  struct ScatterDetails {
    // The dimensions of the corresponding buffer that the scatter slices
    // across. These dimensions must be the major dimensions in the on-device
    // layout of the buffer, and must all be untiled. The scatter acts as if
    // the buffer were transposed/reshaped so that all of these dimensions were
    // combined into a single dimension whose size is the product of the
    // dimensions, and the slice indices correspond to indices in that single
    // combined dimension.
    //
    // For example, if the shape is [3, 4, 128, 128] with [3, 4] as the major
    // dimensions in the layout, and dimensions = {0, 1}, then the buffer is
    // treated as if it were shape [12, 128, 128] and the indices in slices
    // range in [0, 12].
    absl::InlinedVector<int, 3> dimensions;
    // The start and end indices of the slices.
    std::vector<std::pair<int64_t, int64_t>> slices;
  };
  // Each entry in `callbacks` will be called exactly once. As above, in error
  // situations, this may happen before the corresponding entry in
  // `serialaized_descriptors` is fulfilled. This method requires that both
  // `calbacks.size()` and (if Ok) `serialized_descriptors.size()` match the
  // product of the major dimensions specified in `scatter_details`.
  virtual void CopyToRemoteDeviceScattered(
      PjRtFuture<std::vector<std::string>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) = 0;

  // Donates 'this' and returns a new buffer that is ready only when both 'this'
  // and 'dependency' are ready.
  //
  // Once ready, the new buffer's contents will be exactly the contents of
  // 'this'.
  //
  // If either 'this' or 'dependency' transitions to error, then the returned
  // buffer will transition to error.
  virtual absl::StatusOr<std::unique_ptr<PjRtBuffer>>
  DonateWithControlDependency(PjRtFuture<> dependency) {
    return Unimplemented("DonateWithControlDependency is not supported.");
  }

  // Returns a future that can be used to discover when the data in the
  // PjRtBuffer has been computed, or an error has occurred.
  //
  // TODO(b/241967811): change these weird semantics
  // If the buffer has been deleted or donated the returned future will
  // immediately hold an error, however if GetReadyFuture() is called before
  // the buffer has been deleted or donated then the returned future will stay
  // valid (will not transition to error as a consequence of buffer deletion)
  // even if the buffer is subsequently donated or deleted.
  virtual PjRtFuture<> GetReadyFuture() = 0;

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  ABSL_DEPRECATED("Use GetReadyFuture().Await() instead")
  absl::Status BlockHostUntilReady() {
    auto s = GetReadyFuture().Await();
    // Fix up error string because some clients rely on it.
    if (!s.ok() &&
        s.message() == "GetReadyFuture() called on deleted or donated buffer") {
      return InvalidArgument(
          "BlockHostUntilReady() called on deleted or donated buffer");
    }
    return s;
  }

  // Whether this buffer is on CPU and thus allows for certain optimizations.
  virtual bool IsOnCpu() const = 0;
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. If any input/output alias has been specified in
// the computation, the parameter containing the input buffer will be donated
// when passed to the execution.
class PjRtLoadedExecutable : public PjRtExecutable {
 public:
  ~PjRtLoadedExecutable() override = default;

  virtual PjRtClient* client() const = 0;

  virtual const DeviceAssignment& device_assignment() const = 0;

  // Returns named values for cost properties of this executable (such as
  // operations, size of input/outputs, and run time estimate). Properties may
  // differ for different platforms.
  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override;

  // The replica and partition indices of device_assignment to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may not be the
  // case on multi-host platforms. If there are 4 replicas and 2 partitions on a
  // single host platform, size of addressable_device_logical_ids_ is 4*2 = 8.
  struct LogicalDeviceIds {
    int replica;
    int partition;
  };
  virtual absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const = 0;

  // An addressable_device is one which the client can issue commands to.
  // addressable_devices()[i] is the Device to which
  // addressable_device_logical_ids()[i] is assigned.
  virtual absl::Span<PjRtDevice* const> addressable_devices() const = 0;

  // Donation Semantics:
  //
  // The following Execute*() methods will donate the input buffer to the
  // execution if it is specified in the executable. Donation is usually
  // implemented as a transaction: it is acquired in the beginning and committed
  // when the device execution is successully launched. Concurrent donations
  // might either block or return failures.
  //
  // TODO(chky): It is generally desired that concurrent donations do not block,
  // as it otherwise results in deadlock easily. Consider always returning
  // failure on concurrent donations.

  // Executes on devices addressable by the client. Requires executable has a
  // device_assignment and all devices in the device_assignment are addressable
  // by the client.
  //
  // `argument_handles` is `[num_devices, num_args]`.
  //
  // If returned_futures.has_value():
  //   if Execute does not return an error status:
  //     *returned_futures will be resized to be the same length as the return
  //     vector, and each future will become ready once the corresponding device
  //     execute has completed.
  //   else:
  //     *returned_futures is undefined.
  //
  // The caller is *NOT* required to ensure that PjRtLoadedExecutable stays
  // alive until futures are ready.
  virtual absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
          const ExecuteOptions& options,
          std::optional<std::vector<PjRtFuture<>>>& returned_futures) = 0;
  // Convenience wrapper for Execute that never returns futures.
  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options) {
    std::optional<std::vector<PjRtFuture<>>> returned_futures;
    return Execute(std::move(argument_handles), options, returned_futures);
  }

  // Execute the assigned replica/partition on a given `device`. Requires
  // executable has a device_assignment, `device` is present in the
  // device_assignment and addressable by the client.
  //
  // If fill_future is true:
  //   if ExecuteSharded does not return an error status:
  //     returned_future will be filled with a future that will become ready
  //     once the execution has completed.
  //    else:
  //     returned_future will not be modified.
  virtual absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  ExecuteSharded(absl::Span<PjRtBuffer* const> argument_handles,
                 PjRtDevice* device, const ExecuteOptions& options,
                 std::optional<PjRtFuture<>>& returned_future,
                 bool fill_future) = 0;
  // Convenience wrapper for ExecuteSharded that always returns a future.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future) {
    return ExecuteSharded(std::move(argument_handles), device, options,
                          returned_future, /*fill_future=*/true);
  }
  // Convenience wrapper for ExecuteSharded that never returns a future.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) {
    std::optional<PjRtFuture<>> returned_future;
    return ExecuteSharded(std::move(argument_handles), device, options,
                          returned_future, /*fill_future=*/false);
  }

  // Execute on a given `device`. Requires `device` to be addressable by client.
  // Requires executable has exactly 1 replica and 1 partition and no
  // device_assignment (thus portable).
  //
  // If fill_future is true:
  //   if ExecutePortable does not return an error status:
  //     returned_future will be filled with a future that will become ready
  //     once the execution has completed.
  //    else:
  //     returned_future will not be modified.
  virtual absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  ExecutePortable(absl::Span<PjRtBuffer* const> argument_handles,
                  PjRtDevice* device, const ExecuteOptions& options,
                  std::optional<PjRtFuture<>>& returned_future,
                  bool fill_future) = 0;
  // Convenience wrapper for ExecutePortable that always returns a future.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future) {
    return ExecutePortable(std::move(argument_handles), device, options,
                           returned_future, /*fill_future=*/true);
  }
  // Convenience wrapper for ExecutePortable that never returns a future.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) {
    std::optional<PjRtFuture<>> returned_future;
    return ExecutePortable(std::move(argument_handles), device, options,
                           returned_future, /*fill_future=*/false);
  }

  // Asynchronously free resources after the last execution completes.
  virtual void Delete() = 0;

  // True if on-device resources associated with the executable are freed.
  virtual bool IsDeleted() = 0;

  // True if the `returned_futures` output parameter is supported in the
  // Execute*() methods.
  //
  // TODO(b/240696624): Although the PjRt interface require `returned_futures`
  // to be resized correctly if it is not nullopt, some implementation does not
  // implement this. So we have to check whether returned_futures is empty.
  // Remove this method once the implementation is fixed.
  virtual bool IsReturnedFutureSupported() const { return false; }

 protected:
  // Value returned internally from routines that enqueue an execution,
  // combining the result buffers with a future that becomes ready when the
  // execution completes.
  struct Result {
    std::optional<PjRtFuture<>> future;
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  };
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_CLIENT_H_
