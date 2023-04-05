/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_CLIENT_H_

#include <functional>
#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/compiler.h"
#include "tensorflow/compiler/xla/python/ifrt/tuple.h"
#include "tensorflow/compiler/xla/python/ifrt/value.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace ifrt {

using PlatformId = ::xla::PjRtPlatformId;
using ChannelHandle = ::xla::ChannelHandle;

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
  // returned `Future<Status>` for consistency with other IFRT APIs.
  virtual StatusOr<tsl::RCReference<Array>> MakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      std::shared_ptr<const Sharding> sharding, HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) = 0;

  // Builds a larger array out of individual per-device shards.
  virtual StatusOr<tsl::RCReference<Array>> AssembleArrayFromSingleDeviceArrays(
      Shape shape, std::shared_ptr<const Sharding> sharding,
      absl::Span<tsl::RCReference<Array>> arrays,
      ArrayCopySemantics semantics) = 0;

  // Builds a tuple from a sequence of values.
  virtual StatusOr<tsl::RCReference<Tuple>> MakeTuple(
      absl::Span<tsl::RCReference<Value>> values) = 0;

  // The following APIs are taken from `xla::PjRtClient` for fast prototyping.
  // Most of the APIs will be factored out as a `Platform`/`Topology` in the
  // future to facilitate topology discovery and ahead-of-time compilation.

  // TODO(hyeontaek): Remove runtime_type() in favor of LLVM RTTI.
  virtual absl::string_view runtime_type() const = 0;

  // TODO(hyeontaek): Factor them out to a `Platform`/`Topology` class.
  virtual absl::string_view platform_name() const = 0;
  virtual absl::string_view platform_version() const = 0;
  virtual PlatformId platform_id() const = 0;

  virtual int device_count() const = 0;
  virtual int addressable_device_count() const = 0;
  virtual absl::Span<Device* const> devices() const = 0;
  virtual absl::Span<Device* const> addressable_devices() const = 0;
  virtual int process_index() const = 0;

  // TODO(hyeontaek): Consider removing this API. This API is potentially not
  // being used by JAX or will be replaced with explicit device assignment.
  virtual StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const = 0;
  virtual StatusOr<Device*> LookupDevice(int device_id) const = 0;

  virtual StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() = 0;
  virtual StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() = 0;

  // TODO(hyeontaek): Potentially remove this method to encourage supporting
  // only ahead-of-time compilation.
  virtual Compiler* GetDefaultCompiler() = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_CLIENT_H_
