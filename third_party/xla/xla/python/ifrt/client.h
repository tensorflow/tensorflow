/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_CLIENT_H_
#define XLA_PYTHON_IFRT_CLIENT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

using PlatformId = ::xla::PjRtPlatformId;

// TODO(hyeontaek): Generalize DeviceAssignment or hide it from the top-level
// API.
using DeviceAssignment = ::xla::DeviceAssignment;

// Represents an IFRT client. It wraps a runtime that interacts with computation
// devices and memory attached to it.
class Client : public llvm::RTTIExtends<Client, llvm::RTTIRoot> {
 public:
  // Describes the semantics the caller to `MakeArrayFromHostBuffer` expects
  // from the runtime, in a total order from most restrictive to least
  // restrictive.
  //
  // kImmutableOnlyDuringCall:
  // The runtime may not hold references to `data` after the call to
  // `MakeArrayFromHostBuffer` completes. The caller promises that `data` is
  // immutable and will not be freed only for the duration of the
  // `MakeArrayFromHostBuffer` call. `on_done_with_host_buffer` will be called
  // before `MakeArrayFromHostBuffer` returns.

  // kImmutableUntilTransferCompletes:
  // The runtime may hold onto `data` after the call to
  // `MakeArrayFromHostBuffer` returns while the runtime completes transfers to
  // devices. The caller promises not to mutate or free `data` until the
  // transfer completes, at which point the runtime will call
  // `on_done_with_host_buffer`. It is also correct to wait (directly or
  // indirectly) for the `Array`'s ready event. The runtime does not promise a
  // certain ordering between an `on_done_with_host_buffer` call and the
  // `Array`'s ready event.

  // kZeroCopy:
  // The `Array` may alias `data` internally and the runtime may use the `data`
  // contents as long as the buffer is alive. The caller promises to keep `data`
  // alive and not to mutate its contents as long as the buffer is alive; to
  // notify the caller that the buffer may be freed, the runtime will call
  // `on_done_with_host_buffer` when the `Array` is freed. The implementation is
  // free to make a copy and downgrade the semantics to
  // `kImmutableUntilTransferCompletes`. Many non-CPU runtimes will make a copy
  // by default.
  using HostBufferSemantics = ::xla::PjRtClient::HostBufferSemantics;

  // Creates a new array from a host buffer.
  //
  // `data` points to the backing array of the host buffer. Caution:
  // `byte_strides` are allowed to be negative, in which case `data` may need to
  // point to the interior of the buffer, not necessarily its start.
  //
  // If `byte_strides` is omitted, it defaults to a dense layout with dimensions
  // in major-to-minor order. The runtime may return `UNIMPLEMENTED` if
  // `byte_strides` does not equate to a reordering of the dimensions.
  //
  // `on_done_with_host_buffer` is optional and may be null.
  // `on_done_with_host_buffer` will be called iff OK is returned.
  //
  // TODO(hyeontaek): Consider changing `on_done_with_host_buffer` into a
  // returned `Future<absl::Status>` for consistency with other IFRT APIs.
  virtual absl::StatusOr<tsl::RCReference<Array>> MakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      absl::Nonnull<std::shared_ptr<const Sharding>> sharding,
      HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) = 0;

  // Represents a host buffer.
  //
  // TODO(hyeontaek): Consider evolving this structure to `Literal` once it is
  // available in IFRT.
  struct HostBuffer {
    // `data` points to the backing array of the host buffer. Caution:
    // `byte_strides` are allowed to be negative, in which case `data` may need
    // to point to the interior of the buffer, not necessarily its start.
    const void* data;

    DType dtype;
    Shape shape;

    // If omitted, it defaults to a dense layout with dimensions in
    // major-to-minor order. As of 2025, with many IFRT API implementations, API
    // operations that process the HostBuffer return `UNIMPLEMENTED` if asked to
    // process `byte_strides` that do not equate to a reordering of the
    // dimensions.
    //
    // TODO(hyeontaek): Consider generalizing `byte_strides` to a more general
    // layout representation.
    using ByteStrides = std::vector<int64_t>;
    std::optional<ByteStrides> byte_strides;

    // `on_done` is optional and may be null. `on_done` will be called when the
    // host buffer is no longer used by the runtime. It will not be called if
    // the API processing the host buffer has returned an error. For simple and
    // robust cleanup, it is strongly recommended to capture RAII objects in the
    // closure of the callback and leave the callback's function body empty.
    //
    // If a host buffer is used with a zero-copy semantics, the host buffer data
    // should not be accessed by the user until `on_done` is called; the data
    // may be read by the runtime throughout the life of the array created with
    // the host buffer, and it may be even mutated.
    std::function<void()> on_done;
  };

  // Represents the specification of creating an array following an array spec
  // from host buffer shards.
  //
  // `buffers` is a list of pairs of addressable shard indices and a host
  // buffer. `buffers` should include all addressable shards of
  // `array_spec.sharding`.
  //
  // Each host buffer will be used as per-shard data for all addressable shards
  // identified by the shard indices. Host buffers should not require casting or
  // slicing/padding, but may have different layouts (byte strides) from the
  // `array_spec.layout`.
  struct MakeArraysFromHostBufferShardsSpec {
    using ShardIndices = absl::InlinedVector<int64_t, 1>;
    using Buffers = absl::InlinedVector<std::pair<ShardIndices, HostBuffer>, 1>;
    Buffers buffers;
    ArraySpec array_spec;
  };

  // Creates new arrays. For each array, a subset of array shards will be
  // created from a host buffer shard. The resulting array will match the array
  // spec.
  //
  // `specs` may be consumed by the implementation.
  //
  // All resulting arrays should use the same device list and memory kind. i.e.,
  // `specs[i].sharding->devices()` and `specs[i].sharding->memory_kind()` must
  // be equal across all `i`.
  virtual absl::StatusOr<std::vector<tsl::RCReference<Array>>>
  MakeArraysFromHostBufferShards(
      absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
      HostBufferSemantics semantics) = 0;

  // Builds a larger array out of individual per-device shards.
  // TODO(hyeontaek): Replace this API with the version that takes
  // `SingleDeviceShardSemantics` and `dtype`.
  virtual absl::StatusOr<tsl::RCReference<Array>>
  AssembleArrayFromSingleDeviceArrays(
      Shape shape, absl::Nonnull<std::shared_ptr<const Sharding>> sharding,
      absl::Span<tsl::RCReference<Array>> arrays,
      ArrayCopySemantics semantics) = 0;
  virtual absl::StatusOr<tsl::RCReference<Array>>
  AssembleArrayFromSingleDeviceArrays(
      Shape shape, absl::Nonnull<std::shared_ptr<const Sharding>> sharding,
      absl::Span<tsl::RCReference<Array>> arrays,
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) = 0;
  virtual absl::StatusOr<tsl::RCReference<Array>>
  AssembleArrayFromSingleDeviceArrays(
      DType dtype, Shape shape,
      absl::Nonnull<std::shared_ptr<const Sharding>> sharding,
      absl::Span<tsl::RCReference<Array>> arrays,
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) = 0;

  // Copies the arrays to a new set of devices.
  //
  // This method copies individual buffers of each array to the destination
  // devices without altering their physical layout.
  //
  // This API should be used only with arrays that have the same source device
  // list and memory kind. Every IFRT implementation must enforce this by
  // returning an `INVALID_ARGUMENT` error if `arrays` contains different device
  // lists or memory kinds.
  //
  // Implementations may return `UNIMPLEMENTED` if they do not know how to copy
  // the data as instructed.
  //
  // It may fail if the buffer data would be sent from/to an unaddressable
  // device.
  virtual absl::StatusOr<std::vector<tsl::RCReference<Array>>> CopyArrays(
      absl::Span<tsl::RCReference<Array>> arrays,
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind, ArrayCopySemantics semantics) = 0;

  // Remaps shards across input `Array`s to create new `Array`s based on `plan`.
  // This array remapping is a metadata-only operation that can shuffle or
  // extract shards without changing their per-shard interpretation and causing
  // data copy/transfer.
  //
  // There are constraints on `semantics`:
  //
  // * `ArrayCopySemantics::kAlwaysCopy` has an undefined behavior because
  // `RemapArrays` does not copy data.
  // * `ArrayCopySemantics::kReuseInput` is allowed only if the number of inputs
  // is 1. This is safe because each input shard can be used only once.
  // * `ArrayCopySemantics::kDonateInput` is always allowed.
  virtual absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
  RemapArrays(const RemapPlan& plan,
              absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
              ArrayCopySemantics semantics) = 0;

  // Returns a future that becomes ready once all of the values become ready.
  //
  // Timing and error semantics:
  //
  // * The returned future is fulfilled only after all values in `values` become
  //   ready, regardless of their error statuses.
  // * If none of the values have errors, the returned future is fulfilled with
  //   `absl::OkStatus()` once all values are ready.
  // * If there is one or more values with errors, the implementation will pick
  //   one of them arbitrarily to fulfill the returned future.
  //
  // Note: this API currently accepts a span of `tsl::RCReference<Array>` for
  // consistency with other APIs. We may change this to take a span of `Array*`
  // instead to reflect its read-only semantics.
  virtual Future<> GetReadyFuture(
      absl::Span<const tsl::RCReference<Value>> values) = 0;

  // Builds a tuple from a sequence of values.
  virtual absl::StatusOr<tsl::RCReference<Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<Value>> values) = 0;

  // Identifies the IFRT implementation. Most C++ users should use LLVM RTTI to
  // determine the runtime type. This is a string exposed to users mostly for
  // informational reasons.
  virtual absl::string_view runtime_type() const = 0;

  // The following APIs are taken from `xla::PjRtClient` for fast prototyping.
  // Most of the APIs will be factored out as a `Platform`/`Topology` in the
  // future to facilitate topology discovery and ahead-of-time compilation.
  // TODO(hyeontaek): Factor them out to a `Platform`/`Topology` class.
  virtual absl::string_view platform_name() const = 0;
  virtual absl::string_view platform_version() const = 0;
  virtual PlatformId platform_id() const = 0;

  // Returns the attributes of the client. In principle, these try to describe
  // capabilities of a client rather than being a "feature flag".
  //
  // List of officially supported attributes:
  //
  // * supports_executable_serialization (bool; default = true): Whether IFRT
  //   executables produced by this client are serializable. If false, all
  //   executables from this client are considered not serializable.
  virtual const AttributeMap& Attributes() const = 0;

  virtual int device_count() const = 0;
  virtual int addressable_device_count() const = 0;
  virtual absl::Span<Device* const> devices() const = 0;
  virtual absl::Span<Device* const> addressable_devices() const = 0;
  virtual int process_index() const = 0;

  // Returns all devices. The result includes primary devices that are included
  // in `devices()` as well as any other devices that are associated with
  // the primary devices.
  virtual absl::Span<xla::ifrt::Device* const> GetAllDevices() const = 0;

  // TODO(hyeontaek): Consider removing this API. This API is potentially not
  // being used by JAX or will be replaced with explicit device assignment.
  virtual absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const = 0;
  virtual absl::StatusOr<Device*> LookupDevice(DeviceId device_id) const = 0;
  virtual absl::StatusOr<Device*> LookupAddressableDevice(
      int local_hardware_id) const = 0;

  // Creates a device list from the given list of devices.
  virtual DeviceListRef MakeDeviceList(
      absl::Span<Device* const> devices) const = 0;

  // TODO(hyeontaek): Potentially remove this method to encourage supporting
  // only ahead-of-time compilation.
  virtual Compiler* GetDefaultCompiler() = 0;

  // Returns a topology that covers the provided devices.
  virtual absl::StatusOr<std::shared_ptr<Topology>> GetTopologyForDevices(
      const DeviceListRef& devices) const = 0;

  // Returns the default layout on `device` with `memory_kind` for a buffer with
  // `dtype` and single-shard dimensions `dims`.
  // TODO(hyeontaek): Change the API to take `Shape` and `Sharding` instead of
  // single-shard dimensions and device.
  virtual absl::StatusOr<std::shared_ptr<const PjRtLayout>> GetDefaultLayout(
      DType dtype, absl::Span<const int64_t> dims, Device* device,
      xla::ifrt::MemoryKind memory_kind) const = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_CLIENT_H_
