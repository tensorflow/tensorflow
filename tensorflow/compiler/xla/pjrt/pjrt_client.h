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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_CLIENT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"

// API notes:
// PjRt stands for "Pretty much Just another RunTime".

namespace xla {

using PjRtPlatformId = uint64_t;

inline const char* CpuName() {
  static constexpr char kCpuName[] = "cpu";
  return kCpuName;
}
inline const char* GpuName() {
  static constexpr char kGpuName[] = "gpu";
  return kGpuName;
}
inline const char* TpuName() {
  static constexpr char kTpuName[] = "tpu";
  return kTpuName;
}
inline PjRtPlatformId CpuId() {
  static const PjRtPlatformId kCpuId = tensorflow::Fingerprint64(CpuName());
  return kCpuId;
}
inline PjRtPlatformId GpuId() {
  static const PjRtPlatformId kGpuId = tensorflow::Fingerprint64(GpuName());
  return kGpuId;
}
inline PjRtPlatformId TpuId() {
  static const PjRtPlatformId kTpuId = tensorflow::Fingerprint64(TpuName());
  return kTpuId;
}

enum PjRtRuntimeType { kStreamExecutor, kTfrt };
inline constexpr absl::string_view PjRtRuntimeTypeString(PjRtRuntimeType type) {
  switch (type) {
    case kStreamExecutor:
      return "stream_executor";
    case kTfrt:
      return "tfrt";
  }
}

class PjRtClient;

using PjRtDeviceAttribute =
    std::variant<std::string, int64_t, std::vector<int64_t>>;

class PjRtDevice {
 public:
  virtual ~PjRtDevice() {}

  // Return the client that owns this device.
  virtual PjRtClient* client() const = 0;

  // Whether client can issue command to this device.
  virtual bool IsAddressable() const = 0;

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  virtual int id() const = 0;

  // The index of the process that this device belongs to, i.e. is addressable
  // from. This is not always identical to PjRtClient::process_index() in a
  // multi-process setting, where each client can see devices from all
  // processes, but only a subset of them are addressable and have the same
  // process_index as the client.
  virtual int process_index() const = 0;

  // Opaque hardware ID, e.g., the CUDA device number, useful for identifying
  // which GPU when interacting with non-JAX code. In general, not guaranteed to
  // be dense, and -1 if undefined.
  virtual int local_hardware_id() const = 0;

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual absl::string_view device_kind() const = 0;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  virtual absl::string_view DebugString() const = 0;

  // Debug string suitable for reading by end users, should be reasonably terse,
  // for example: "CpuDevice(id=0)".
  virtual absl::string_view ToString() const = 0;

  // Returns a scoped event that the caller uses to tell the PjRtClient that
  // there is asynchronous work happening that depends on activity on the
  // PjRtDevice. See comment on class definition in pjrt_future.h.
  //
  // Only some PjRtDevice implementations support ScopedAsyncTrackingEvent, and
  // those that do not will return nullptr.
  virtual std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const = 0;

  // Transfer the given literal to the infeed queue.
  virtual Status TransferToInfeed(const LiteralSlice& literal) = 0;

  // Transfer and return a value of the given shape from the outfeed queue.
  virtual Status TransferFromOutfeed(MutableBorrowingLiteral literal) = 0;

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the PjRtDevice.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const = 0;
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
using PjRtCrossHostSendCancelNotifier =
    std::function<void(absl::string_view serialized_descriptor, Status reason,
                       std::function<void(Status)> on_canceled)>;
// State asynchronously returned by MakeCrossHostReceiveBuffers. "descriptors"
// will match the returned PjRtBuffer objects 1:1. Specifically, each PjRtBuffer
// returned by MakeCrossHostReceiveBuffers will have one
// PjRtCrossHostRecvDescriptors object containing it descriptor(s).
struct PjRtCrossHostRecvState {
  std::vector<PjRtCrossHostRecvDescriptors> descriptors;
  PjRtCrossHostSendCancelNotifier cancel_notifier;
};
using PjRtCrossHostRecvNotifier =
    std::function<void(StatusOr<PjRtCrossHostRecvState>)>;

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
  explicit CopyToDeviceStream(int64_t total_bytes, int64_t granule_bytes)
      : total_bytes_(total_bytes), granule_bytes_(granule_bytes) {}

  // Emplaces a new Chunk of data to copy to the device. Returns a non-OK status
  // if the Chunk's size causes the amount of transferred data to exceed
  // total_bytes() or if the stream is already complete.
  //
  // The size of the chunk must be a multiple of granule_bytes().
  // TODO(jmolloy): Enforce the granule size.
  Status AddChunk(PjRtChunk chunk);

  // Returns the total amount of data the stream expects to be transferred.
  int64_t total_bytes() const { return total_bytes_; }

  // Returns the granule size in bytes. The size of the chunk added to this
  // stream must be a multiple of this number.
  int64_t granule_size_in_bytes() const { return granule_bytes_; }

  // Returns the amount of data the stream currently has either transferred or
  // has buffered to transfer.
  int64_t current_bytes() const {
    absl::MutexLock lock(&mu_);
    return current_bytes_;
  }

  // Returns true if the stream is complete; all expected bytes have been
  // transferred or are buffered to transfer.
  bool IsComplete() const {
    absl::MutexLock lock(&mu_);
    return current_bytes_ == total_bytes_;
  }

  // Returns true if the stream is empty; no data has been queued.
  bool empty() const { return current_bytes() == 0; }

  // Consumes the next chunk. If no chunks remain, returns nullopt. Blocks
  // until a chunk is available.
  std::optional<PjRtChunk> ConsumeNextChunk();

  // Members are protected to allow subclassing for mocking in tests.
 protected:
  int64_t total_bytes_;
  int64_t granule_bytes_;
  int64_t current_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  std::deque<PjRtChunk> buffered_chunks_ ABSL_GUARDED_BY(mu_);
  mutable absl::Mutex mu_;
};

class PjRtHostMemoryForDeviceManager {
 public:
  virtual ~PjRtHostMemoryForDeviceManager();

  // Transforms the host memory representations of a shape with the host layout
  // to the host memory representation of the same shape with the device layout.
  // `src_shape` and `dst_shape` may only differ in their layouts.
  virtual StatusOr<PjRtChunk> ToDeviceLayout(const void* src_data,
                                             size_t src_size,
                                             const Shape& host_shape,
                                             const Shape& device_shape) = 0;

  // Transforms the host memory representations of a shape with the device
  // layout to the host memory representation of the same shape with the host
  // layout. `src_shape` and `dst_shape` may only differ in their layouts.
  virtual Status ToHostLayout(const void* src_data, size_t src_size,
                              const Shape& src_shape, void* dst_data,
                              size_t dst_size, const Shape& dst_shape) = 0;
};

class PjRtLoadedExecutable;

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
  virtual StatusOr<PjRtDevice*> LookupDevice(int device_id) const = 0;

  // Return an addressable PjRtDevice for a given
  // PjRtDevice::local_hardware_id().
  virtual StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const = 0;

  // Return an ID that identifies the platform (CPU/GPU/TPU).
  virtual PjRtPlatformId platform_id() const = 0;

  // Returns a string that identifies the platform (CPU/GPU/TPU).
  virtual absl::string_view platform_name() const = 0;

  // Returns a string containing human-readable, platform-specific version info
  // (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
  virtual absl::string_view platform_version() const = 0;

  // TODO(b/244756954): Rethink this function altogether
  // Returns an enum that identifies the type of runtime being used under this
  // client.
  virtual PjRtRuntimeType runtime_type() const = 0;

  // Return a device-specific default device assignment, e.g., GPU and TPU may
  // be different.
  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const = 0;

  // Returns a device-specific default device assignment for multi-slice system.
  // If num_replicas_per_slice is not defined (nullopt) then we assume that
  // all the partitions live entirely on a single slice and that all cross slice
  // communication happens across replicas assuming then that
  // num_replicas_per_slice is going to be "num_replicas / num_slices".
  // TODO(zhangqiaorjc): Convert this to pure virtual and push down.
  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, std::optional<int> num_replicas_per_slice,
      int num_partitions, const MultiSliceConfig* multi_slice_config) const {
    return Unimplemented("Multi slice device assignment is not supported.");
  }

  // Returns a backend-specific HLO cost analysis visitor.
  virtual StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() = 0;

  // Compile `computation` with given `options`.
  virtual StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) = 0;

  // Variant of `Compile` that accepts an MLIR module.
  virtual StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) = 0;

  // Generates a unique fingerprint for `executable`, may be std::nullopt.
  virtual StatusOr<std::optional<std::string>> ExecutableFingerprint(
      const PjRtLoadedExecutable& executable) const = 0;

  // Returns a platform-specific serialization of `executable`. The
  // serialization is not guaranteed to be stable over time. `executable` must
  // have been produced by this client.
  virtual StatusOr<std::string> SerializeExecutable(
      const PjRtLoadedExecutable& executable) const = 0;

  // Deserializes a serialized executable as produced by
  // SerializeExecutable(). `serialized` must have been produced by a client of
  // the same platform and version as this one.
  virtual StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized, CompileOptions options) = 0;

  // LoadSerializedExecutable takes the serialized output of PjRtExecutable. The
  // returned executable is loaded by this client. The same checks are made as
  // in Load that the serialized executable is compatible with the client.
  // LoadSerializedExecutable will materialize CompileOptions from within the
  // serialized executable unlike 'DeserializeExecutable' above that accepts
  // CompileOptions.
  virtual StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(absl::string_view serialized) const {
    return Unimplemented("Loading serialized executable not supported.");
  }

  // Loads the executable returns aa PjRtLoadedExecutable runnable by this
  // client. Returns an error if the PjRtExecutable was created with an
  // incompatible topology or client.
  // PjRtExecutable contains a copy of the CompileOptions that was used to
  // generate the executable. Load will use the CompileOptions from within the
  // executable.
  virtual StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::unique_ptr<PjRtExecutable> executable) {
    return Unimplemented("Loading executable not supported.");
  }

  // Creates a buffer on the device without initializing or copying any data.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) = 0;

  // A client may want to create a buffer, and hand the buffer to other PjRt
  // methods, before the data to store in the buffer is available to the client.
  // This is supported using CreateBuffersForAsyncTransfer, which returns an
  // AsyncBufferTransferManager helper object.
  //
  // The PjRtBuffers can be retrieved from the AsyncBufferTransferManager and
  // safely passed immediately to downstream PjRt method calls. Subsequently the
  // client can call methods on the AsyncBufferTransferManager object to copy
  // data into the buffers, and once the data copies are complete, the buffers'
  // definition events will automatically become ready, unblocking downstream
  // consumers of the buffers.
  //
  // A single call to CreateBuffersForAsyncTransfer creates a "batch" of buffers
  // that share a single definition event, which may amortize some performance
  // overheads, but means that none of the buffers are available to downstream
  // consumers until all the transfers have completed. Multiple calls to
  // CreateBuffersForAsyncTransfer should be made if it is desirable for buffers
  // to become available as soon as transfers into them complete.

  // Helper class to all clients to asynchronously transfer data into buffers
  // that are created uninitialized, see comments immediately above.
  class AsyncBufferTransferManager {
   public:
    virtual ~AsyncBufferTransferManager() = default;

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
    virtual Status TransferLiteralToBuffer(int buffer_index,
                                           const LiteralSlice& literal,
                                           std::function<void()> on_done) = 0;

    // Returns the on-device size in bytes of buffer buffer_index.
    virtual size_t buffer_size(int buffer_index) const = 0;

    // Transfers 'data' into buffer_index. 'data' must be already laid out in
    // the correct on-device format, for example returned by a call to
    // buffer->CopyRawToHost. No transfer calls into buffer_index can be made
    // after this call. on_done is called when the transfer is complete but
    // before the buffers are made available to their consumers. 'data' must
    // remain in scope until on_done is called.
    virtual Status TransferRawDataToBuffer(int buffer_index,
                                           absl::string_view data,
                                           std::function<void()> on_done) = 0;

    // Transfers 'data' into a sub-buffer of buffer_index starting at offset, of
    // length transfer_size. 'data' must be already laid out in the correct
    // on-device format, for example returned by a call to
    // buffer->CopyRawToHost. If is_last_transfer is false then the buffer
    // remains unavailable to consumers after the transfer completes. If
    // is_last_transfer is true then the buffer becomes available to consumers
    // after the transfer completes, and no transfer calls into buffer_index can
    // be made after this call. on_done is called when the transfer is complete
    // but before the buffers are made available to their consumers. 'data' must
    // remain in scope until on_done is called.
    virtual Status TransferRawDataToSubBuffer(
        int buffer_index, const void* data, int64_t offset,
        int64_t transfer_size, bool is_last_transfer,
        std::function<void()> on_done) = 0;

    // Indicates that a client error occurred and the transfers will never
    // complete. Puts all buffers in an error state. For the stream executor
    // client, since error states are not well supported, this triggers a fatal
    // error.
    //
    // SetTransferError may be called at most once, and may not be called unless
    // at least one buffer has not yet had its final transfer initiated.
    virtual void SetTransferError(Status error) = 0;

    // Adds the specified key/value metadata for the transfer operation.
    // This is typically used for debugging purposes, such as adding a handle
    // that can be used to identify transfer operations.
    using TransferMetadata = absl::flat_hash_map<std::string, std::string>;
    virtual void AddTransferMetadata(const TransferMetadata& metadata) = 0;
  };

  // Returns a manager for async transfers into a set of buffers with on-host
  // shapes 'shapes'.
  virtual StatusOr<std::unique_ptr<AsyncBufferTransferManager>>
  CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                PjRtDevice* device) = 0;

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
    // `data` contents as long as the buffer is alive. The caller promises to
    // keep `data` alive and not to mutate its contents as long as the buffer is
    // alive; to notify the caller that the buffer may be freed, the runtime
    // will call `on_done_with_host_buffer` when the PjRtBuffer is freed. On
    // non-CPU platforms this acts identically to
    // kImmutableUntilTransferCompletes.
    kZeroCopy,
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
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer, PjRtDevice* device) = 0;

  // Note that literal must remain in scope until the transfer has completed, so
  // the caller should, for example, wait for GetReadyFuture().Await()
  // completes on the return value before letting literal go out of scope.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) = 0;

  // Creates a PjRtBuffer that is a non-owned view of an on-device
  // buffer (typically allocated by another library).
  // on_delete_callback is called when the PjRtBuffer is done with the on-device
  // buffer. The buffer may be mutated, for example, if the buffer is donated
  // to an Execute operation.
  // TODO(phawkins): Currently this API assumes the buffer is ready to use
  // immediately on the device. Extend it to support, for example, waiting for a
  // CUDA stream/event.
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) = 0;

  // Returns platform-dependent address for the given buffer that is often but
  // not guaranteed to be the physical/device address.
  virtual StatusOr<std::uintptr_t> UnsafeBufferPointer(PjRtBuffer* buffer);

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
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) = 0;

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
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) = 0;

  // Create ChannelHandles for XLA send/recv.
  virtual StatusOr<ChannelHandle> CreateChannelHandle() = 0;
  virtual StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() = 0;
  virtual StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() = 0;

  // TODO(zhangqiaorjc): Experimental API to be removed.
  // Defragment device memory.
  virtual Status Defragment() = 0;

  // Return the PjRtHostMemoryForDeviceManager for this client. It can be
  // nullptr if the implementation does not provide one.
  PjRtHostMemoryForDeviceManager* GetPjRtHostMemoryForDeviceManager() const {
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

  virtual const Shape& on_device_shape() const = 0;

  // Same as on_device_shape when the shape is static. When the shape is
  // dynamic, it gathers the metadata from the device and returns a static shape
  // representing the logical shape of the data. This approach is identical to
  // how tensorflow and xrt setup the output buffer in the graph.
  //
  // Since this method actually acquires locks and communicate with the device,
  // it does not have the const qualifier, similar to what ToLiteral does.
  virtual StatusOr<Shape> logical_on_device_shape() = 0;
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

   protected:
    void* data_ptr_;
  };
  virtual StatusOr<std::unique_ptr<ExternalReference>>
  AcquireExternalReference() = 0;

  // Asynchronously copies the buffer's value into `literal`.
  //
  // Return value is a future the caller can use to discover when the copy has
  // completed. The transfer respects the layout of `literal`; to specify a
  // particular layout, set the layout before calling `ToLiteral`.
  virtual PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) = 0;

  // Copies the buffer's value into `literal`. Calls `on_ready` when the value
  // (or an error) is ready. The transfer respects the layout of `literal`; to
  // specify a particular layout, set the layout before calling `ToLiteral`.
  ABSL_DEPRECATED("Use ToLiteral(...).OnReady() instead")
  void ToLiteral(MutableLiteralBase* literal,
                 std::function<void(Status)> on_ready) {
    ToLiteral(literal).OnReady(std::move(on_ready));
  }

  // Synchronous overload of ToLiteral, as a convenience.
  Status ToLiteralSync(MutableLiteralBase* literal) {
    absl::Notification done;
    Status status;
    ToLiteral(literal, [&](Status s) {
      status = std::move(s);
      done.Notify();
    });
    done.WaitForNotification();
    return status;
  }

  // Convenience synchronous overload that allocates a literal with a default
  // layout.
  StatusOr<std::shared_ptr<Literal>> ToLiteralSync() {
    auto literal = std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(on_device_shape()));
    TF_RETURN_IF_ERROR(ToLiteralSync(literal.get()));
    return literal;
  }

  // Returns the number of bytes of the buffer storage on the device.
  virtual StatusOr<size_t> GetOnDeviceSizeInBytes() const = 0;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. The
  // returned future transitions to ready on error, or after the transfer has
  // completed.
  virtual PjRtFuture<Status> CopyRawToHost(void* dst, int64_t offset,
                                           int64_t transfer_size) = 0;

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
  virtual StatusOr<std::unique_ptr<ExternalReference>>
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
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) = 0;

  // Copies the buffer to the remote device encoded in serialized_descriptor.
  // This call must be preceded by a call to MakeCrossHostReceiveBuffers on the
  // remote host's destination device. MakeCrossHostReceiveBuffers takes an
  // array of shapes to construct the destination buffers, and a callback
  // supplies an array containing both the destination buffers, and a serialized
  // descriptor for each buffer. For each destination buffer there should be a
  // matching call to src->CopyToRemoteDevice on a remote host for a src buffer
  // of the corresponding shape. serialized_descriptor is the string returned by
  // the callback along with the corresponding destination buffer.
  //
  // When the send either completes or fails, on_done will be called. If
  // status is Ok then it is guaranteed that sends_were_enqueued==true.
  // Otherwise, if sends_were_enqueued==false then the sender should contact
  // the receiver out of band to request cancellation of the transfer. If
  // !status.ok() and sends_were_enqueued==true then it is not possible to
  // determine whether the transfer succeeded and the system is in an
  // undefined state. This undefined state almost certainly indicates an
  // unrecoverable hardware error.
  //
  // See note on semantics of cross-device copies in the class definition
  // comment for PjRtClient.
  using RemoteSendCallback =
      std::function<void(Status status, bool sends_were_enqueued)>;
  virtual void CopyToRemoteDevice(absl::string_view serialized_descriptor,
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
  virtual void CopyToRemoteDeviceScattered(
      absl::Span<const std::pair<std::string, RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const ScatterDetails& scatter_details) = 0;

  // Returns a future that can be used to discover when the data in the
  // PjRtBuffer has been computed, or an error has occurred.
  //
  // TODO(b/241967811): change these weird semantics
  // If the buffer has been deleted or donated the returned future will
  // immediately hold an error, however if GetReadyFuture() is called before
  // the buffer has been deleted or donated then the returned future will stay
  // valid (will not transition to error as a consequence of buffer deletion)
  // even if the buffer is subsequently donated or deleted.
  virtual PjRtFuture<Status> GetReadyFuture() = 0;

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  ABSL_DEPRECATED("Use GetReadyFuture().Await() instead")
  Status BlockHostUntilReady() {
    auto s = GetReadyFuture().Await();
    // Fix up error string because some clients rely on it.
    if (!s.ok() && s.error_message() ==
                       "GetReadyFuture() called on deleted or donated buffer") {
      return InvalidArgument(
          "BlockHostUntilReady() called on deleted or donated buffer");
    }
    return s;
  }

  // Calls callback when the buffer is ready.
  //
  //   buf->OnReady(callback);
  //
  // is semantically almost identical to:
  //
  //   ForkThread([]() { callback(buf->Await()); });
  //
  // the only difference being that the callback may happen immediately on the
  // calling thread. (The implementation may also be more efficient.)
  //
  // The interface makes no assumptions about what thread calls callback, so the
  // caller must ensure that callback returns quickly and hands off long-running
  // work or any blocking operation to a caller-managed threadpool.
  ABSL_DEPRECATED("Use GetReadyFuture().OnReady() instead")
  void OnReady(std::function<void(Status)> callback) {
    return GetReadyFuture().OnReady(std::move(callback));
  }

  // Whether this buffer is on CPU and thus allows for certain optimizations.
  virtual bool IsOnCpu() const = 0;
};

class ExecuteContext {
 public:
  virtual ~ExecuteContext() = default;
};

struct PjRtTransferMetadata {
  Shape device_shape;
};

struct SendCallback {
  int64_t channel_id;
  // The callback for retrieving the send value. It will be invoked once for
  // each invocation of the corresponding Send op in the HLO program (So it can
  // be invoked multiple times if it is in a loop). Currently there is no
  // guarantee that the callback here will be invoked in the same order as their
  // corresponding HLO Send ops. The callback can also return errors to indicate
  // the execution should fail.
  //
  // IMPORTANT: the implementation might NOT signal the error to the execution,
  // and the execution will run to completion with UNDEFINED DATA returned by
  // the callback. If there is any potential control flow that depends on the
  // value of the returned data, an error return is unsafe.
  //
  // TODO(chky): Currently the callback invocation order may not be consistent
  // with the HLO send op invocation order, due to limitations in some PjRt
  // implementation. Consider making it strictly the same order as HLO program.
  std::function<Status(const PjRtTransferMetadata& metadata, PjRtChunk chunk,
                       size_t total_size_in_bytes, bool done)>
      callback;
};

struct RecvCallback {
  int64_t channel_id;
  // The callback for feeding the recv value. It will be invoked once for each
  // invocation of the corresponding Recv op in the HLO program (So it can be
  // invoked multiple times if it is in a loop). Currently there is no
  // guarantee that the callback here will be invoked in the same order as their
  // corresponding HLO Recv ops.
  std::function<void(const PjRtTransferMetadata& metadata,
                     CopyToDeviceStream& stream)>
      callback;
};

struct ExecuteOptions {
  // If true, the client must pass a single PjRtBuffer which contains all of
  // the arguments as a single XLA tuple, otherwise each argument must be
  // passed in its own PjRtBuffer. May only be true if the executable was
  // compiled with parameter_is_tupled_arguments==true.
  bool arguments_are_tupled = false;
  // If true, the computation must return a tuple, which will be destructured
  // into its elements.
  bool untuple_result = false;
  // If non-zero, identifies this execution as part of a potentially
  // multi-device launch. This can be used to detect scheduling errors, e.g. if
  // multi-host programs are launched in different orders on different hosts,
  // the launch IDs may be used by the runtime to detect the mismatch.
  int32_t launch_id = 0;
  // If non-null, an opaque context passed to an execution that may be used to
  // supply additional arguments to a derived class of PjRtExecutable.
  const ExecuteContext* context = nullptr;
  // If true, check that the PjRtBuffer argument shapes match the compiled
  // shapes. Otherwise, any shape with the right size on device may be passed.
  bool strict_shape_checking = true;

  // Set multi_slice_config when the computation spans multiple slices. The
  // config should match what was used during compilation to generate this
  // executable.
  const MultiSliceConfig* multi_slice_config = nullptr;

  // The send/recv callbacks for PjRt execution. The first level span is for
  // multi-device parallel execution, the second level vector contains the
  // callbacks for all send/recv ops in the executable. These callbacks can be
  // stateful and the user code is responsible for managing the states here.
  // These callbacks must outlive the execution.
  absl::Span<const std::vector<SendCallback>> send_callbacks;
  absl::Span<const std::vector<RecvCallback>> recv_callbacks;

  // The `execution_mode` decides whether the execution will be invoked in the
  // caller thread or launched to a separate thread. By default, the
  // implementation may choose either strategy or use a heuristic to decide.
  // Currently it is only applied to CPU implementations
  enum class ExecutionMode { kDefault = 0, kSynchronous, kAsynchronous };
  ExecutionMode execution_mode = ExecutionMode::kDefault;
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. If any input/output alias has been specified in
// the computation, the parameter containing the input buffer will be donated
// when passed to the execution.
class PjRtLoadedExecutable : public PjRtExecutable {
 public:
  virtual ~PjRtLoadedExecutable() = default;

  virtual PjRtClient* client() const = 0;

  virtual const DeviceAssignment& device_assignment() const = 0;

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
  // implemented as a transaction: it is acquired in the begining and committed
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
  virtual StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
          const ExecuteOptions& options,
          std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) = 0;
  // Convenience wrapper for Execute that never returns futures.
  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options) {
    std::optional<std::vector<PjRtFuture<Status>>> returned_futures;
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
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) = 0;
  // Convenience wrapper for ExecuteSharded that always returns a future.
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future) {
    return ExecuteSharded(std::move(argument_handles), device, options,
                          returned_future, /*fill_future=*/true);
  }
  // Convenience wrapper for ExecuteSharded that never returns a future.
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) {
    std::optional<PjRtFuture<Status>> returned_future;
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
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) = 0;
  // Convenience wrapper for ExecutePortable that always returns a future.
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<Status>>& returned_future) {
    return ExecutePortable(std::move(argument_handles), device, options,
                           returned_future, /*fill_future=*/true);
  }
  // Convenience wrapper for ExecutePortable that never returns a future.
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) {
    std::optional<PjRtFuture<Status>> returned_future;
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
    std::optional<PjRtFuture<Status>> future;
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  };
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_CLIENT_H_
