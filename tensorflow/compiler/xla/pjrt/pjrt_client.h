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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

// API notes:
// PjRt stands for "Pretty much Just another RunTime".

namespace xla {

using PjRtPlatformId = uint64;

constexpr char kCpuName[] = "cpu";
constexpr char kGpuName[] = "gpu";
constexpr char kTpuName[] = "tpu";
static const PjRtPlatformId kCpuId = tensorflow::Fingerprint64(kCpuName);
static const PjRtPlatformId kGpuId = tensorflow::Fingerprint64(kGpuName);
static const PjRtPlatformId kTpuId = tensorflow::Fingerprint64(kTpuName);

enum PjRtRuntimeType { kStreamExecutor, kTfrt };
static constexpr absl::string_view PjRtRuntimeTypeString(PjRtRuntimeType type) {
  switch (type) {
    case kStreamExecutor:
      return "stream_executor";
    case kTfrt:
      return "tfrt";
  }
}

class PjRtClient;

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

  // Deprecated; please switch to process_index().
  int task_id() const { return process_index(); }

  // Opaque hardware ID, e.g., the CUDA device number, useful for identifying
  // which GPU when interacting with non-JAX code. In general, not guaranteed to
  // be dense, and -1 if undefined.
  virtual int local_hardware_id() const = 0;

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual absl::string_view device_kind() const = 0;

  virtual std::string DebugString() const = 0;

  // Transfer the given literal to the infeed queue.
  virtual Status TransferToInfeed(const LiteralSlice& literal) = 0;

  // Transfer and return a value of the given shape from the outfeed queue.
  virtual Status TransferFromOutfeed(MutableBorrowingLiteral literal) = 0;
};

// Forward declaration.
class PjRtBuffer;

// Helper struct for cross host transfers, returned by the callback from a call
// to PjRtBuffer::MakeCrossHostReceiveBuffers or
// PjRtBuffer::MakeCrossHostReceiveBuffersForGather.
struct PjRtCrossHostRecvBuffer {
  // There is one serialized_descriptor per sub-buffer being gathered (i.e. a
  // single descriptor if the buffer is returned from a call to
  // MakeCrossHostReceiveBuffers). The descriptor should be transmitted to the
  // sender(s) and passed to a call to src_buffer->CopyToRemoteDevice.
  absl::InlinedVector<std::string, 1> serialized_descriptors;
  // The buffer that will hold the result of the transfer.
  std::unique_ptr<PjRtBuffer> buffer;
};
using PjRtCrossHostRecvNotifier =
    std::function<void(StatusOr<std::vector<PjRtCrossHostRecvBuffer>>&&)>;

struct CompileOptions {
  // The layouts of the arguments that the computation should expect.
  absl::optional<std::vector<Shape>> argument_layouts;

  // If true, the supplied computation expects its arguments to be wrapped in a
  // tuple and passed as a single parameter.
  bool parameter_is_tupled_arguments = false;

  // XLA's compilation time options.
  ExecutableBuildOptions executable_build_options;

  // If true, the executable can be run on any device. May only be true if
  // !executable_build_options.has_device_assignment(), so only applies to
  // single-device executables. Beware: on GPUs, sometimes an executable
  // compiled for one device doesn't run on another.
  bool compile_portable_executable = false;
};

class PjRtExecutable;

// Encapsulates the state of Python session with XLA.
//
// It is the responsibility of the client of this API to keep the PjRtClient
// alive as long as any of the other runtime objects are alive.
class PjRtClient {
 public:
  virtual ~PjRtClient() = default;

  // Return the process index of this client. Always 0 in single-process
  // settings.
  virtual int process_index() const = 0;

  // Deprecated; please switch to process_index().
  int task_id() const { return process_index(); }

  // Return the number of devices in the entire computation. In multi-headed
  // client setting, some are addressable by this client, some are not. In a
  // single-client setting, this is equal to the number of addressable devices.
  virtual int device_count() const = 0;

  // Return number of addressable devices. Addressable devices are those that
  // the client can issue commands to.
  virtual int addressable_device_count() const = 0;

  // Return all devices in the entire computation, including addressable and
  // non-addressable devices.
  virtual absl::Span<PjRtDevice* const> devices() const = 0;

  // Return only addressable devices.
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

  // Returns an enum that identifies the type of runtime being used under this
  // client.
  virtual PjRtRuntimeType runtime_type() const = 0;

  // Return a device-specific default device assignment, e.g., GPU and TPU may
  // be different.
  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const = 0;

  // Returns a backend-specific HLO cost analysis visitor.
  virtual StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() = 0;

  // Compile `computation` with given `options`.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) = 0;

  // Generates a unique fingerprint for `executable`, may be absl::nullopt.
  virtual StatusOr<absl::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const = 0;

  // Returns a platform-specific serialization of `executable`. The
  // serialization is not guaranteed to be stable over time. `executable` must
  // have been produced by this client.
  virtual StatusOr<std::string> SerializeExecutable(
      const PjRtExecutable& executable) const = 0;

  // Deserializes a serialized executable as produced by
  // SerializeExecutable(). `serialized` must have been produced by a client of
  // the same platform and version as this one.
  virtual StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized, std::unique_ptr<HloModule> hlo_module,
      CompileOptions options) = 0;

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
        int buffer_index, const void* data, int64 offset, int64 transfer_size,
        bool is_last_transfer, std::function<void()> on_done) = 0;

    // Indicates that a client error occurred and the transfers will never
    // complete. Puts all buffers in an error state. For the stream executor
    // client, since error states are not well supported, this triggers a fatal
    // error.
    //
    // SetTransferError may be called at most once, and may not be called unless
    // at least one buffer has not yet had its final transfer initiated.
    virtual void SetTransferError(Status error) = 0;
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
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, const Shape& shape,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer, PjRtDevice* device) = 0;

  // Note that literal must remain in scope until the transfer has completed, so
  // the caller should, for example, wait for BlockHostUntilReady() completes on
  // the return value before letting literal go out of scope.
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

  // Asynchronously makes a vector of PjRtBuffers that can be used to receive
  // cross host transfers using `client` on `device'. `shapes` must be the exact
  // shapes, with identical layouts, corresponding to the buffers that will be
  // sent. When resources for the transfer are available, notifier will be
  // called with a vector of PjRtCrossHostRecvBuffer structs, one for each
  // shape in `shapes`. Each struct contains a buffer that will contain the
  // received value, and an opaque string that should be transmitted to the
  // sending host and used in a call to CopyToRemoteDevice. None of the recv
  // buffers will become ready until *all* of the sends have completed.
  virtual void MakeCrossHostReceiveBuffers(
      absl::Span<const Shape> shapes, PjRtDevice* device,
      PjRtCrossHostRecvNotifier&& notifier) = 0;

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
    std::vector<int64> slice_boundaries;
  };
  virtual void MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier&& notifier) = 0;

  // Create ChannelHandles for XLA send/recv.
  virtual StatusOr<ChannelHandle> CreateChannelHandle() = 0;
  virtual StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() = 0;
  virtual StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() = 0;

  // TODO(zhangqiaorjc): Experimental API to be removed.
  // Defragment device memory.
  virtual Status Defragment() = 0;
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

  // Copies the buffer's value into `literal`. Calls `on_ready` when the value
  // (or an error) is ready. The transfer respects the layout of `literal`; to
  // specify a particular layout, set the layout before calling `ToLiteral`.
  virtual void ToLiteral(MutableLiteralBase* literal,
                         std::function<void(Status)> on_ready) = 0;

  // Synchronous overload of ToLiteral, as a convenience.
  Status ToLiteral(MutableLiteralBase* literal) {
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
  StatusOr<std::shared_ptr<Literal>> ToLiteral() {
    auto literal = std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(on_device_shape()));
    TF_RETURN_IF_ERROR(ToLiteral(literal.get()));
    return literal;
  }

  // Returns the number of bytes of the buffer storage on the device.
  virtual StatusOr<size_t> GetOnDeviceSizeInBytes() const = 0;

  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than GetOnDeviceSizeInBytes. on_ready
  // is called if and only if CopyRawToHost returns OK. on_ready will be called
  // with a non-OK status if the buffer asynchronously transitions to an error
  // state.
  virtual Status CopyRawToHost(void* dst, int64 offset, int64 transfer_size,
                               std::function<void(Status)> on_ready) = 0;

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
  // BlockHostUntilReady before calling ReleaseDeviceMemoryOwnership with
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
  virtual Status CopyToRemoteDevice(
      absl::string_view serialized_descriptor) = 0;
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
    std::vector<std::pair<int64, int64>> slices;
  };
  virtual Status CopyToRemoteDeviceScattered(
      absl::Span<const std::string> serialized_descriptors,
      const ScatterDetails& scatter_details) = 0;

  // Blocks the host until the buffer's value has been computed and is ready for
  // immediate use on the device. Useful in particular for timing benchmarks.
  virtual Status BlockHostUntilReady() = 0;

  // Whether this buffer is on CPU and thus allows for certain optimizations.
  virtual bool IsOnCpu() const = 0;
};

class ExecuteContext {
 public:
  virtual ~ExecuteContext() = default;
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
  int32 launch_id = 0;
  // If non-null, an opaque context passed to an execution that may be used to
  // supply additional arguments to a derived class of PjRtExecutable.
  const ExecuteContext* context = nullptr;
  // If true, check that the PjRtBuffer argument shapes match the compiled
  // shapes. Otherwise, any shape with the right size on device may be passed.
  bool strict_shape_checking = true;
};

// Represents a compiled computation that can be executed given handles to
// device-allocated literals. If any input/output alias has been specified in
// the computation, the parameter containing the input buffer will be donated
// when passed to the execution.
class PjRtExecutable {
 public:
  virtual ~PjRtExecutable() = default;

  virtual PjRtClient* client() const = 0;

  // Unique name for this executable, e.g., HloModule name.
  virtual absl::string_view name() const = 0;

  virtual int num_replicas() const = 0;

  virtual int num_partitions() const = 0;

  virtual int64 SizeOfGeneratedCodeInBytes() const = 0;

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

  // Return an HloModule (optimized) per partition.
  virtual StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const = 0;

  // Executes on devices addressable by the client. Requires executable has a
  // device_assignment and all devices in the device_assignment are addressable
  // by the client.
  // `argument_handles` is `[num_devices, num_args]`.
  virtual StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
          const ExecuteOptions& options) = 0;

  // Execute the assigned replica/partition on a given `device`. Requires
  // executable has a device_assignment, `device` is present in the
  // device_assignment and addressable by the client.
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) = 0;

  // Execute on a given `device`. Requires `device` to be addressable by client.
  // Requires executable has exactly 1 replica and 1 partition and no
  // device_assignment (thus portable).
  virtual StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options) = 0;

  // Asynchronously free resources after the last execution completes.
  virtual void Delete() = 0;

  // True if on-device resources associated with the executable are freed.
  virtual bool IsDeleted() = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_CLIENT_H_
