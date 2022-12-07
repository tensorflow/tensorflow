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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_ARRAY_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "tensorflow/compiler/xla/python/ifrt/future.h"
#include "tensorflow/compiler/xla/python/ifrt/shape.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace ifrt {

class Client;

// Semantics for operations that may copy or move sharded buffers in an array.
enum class ArrayCopySemantics : int {
  // Always creates new buffers to construct an output array. Mutation of the
  // output array buffers will not mutate the input array buffers.
  kAlwaysCopy = 0,

  // Tries to use the existing buffers of the input array to construct an output
  // array. In-place mutation of the output array buffers may also mutate the
  // input array buffers.
  kReuseInput,

  // Tries to use existing buffers of the input array to construct an output
  // array. Drops the ownership of unused buffers in the input array, making the
  // input array no longer usable and reclaiming its on-device resources.
  kDonateInput,
};

// Represents a single logical array from one or more sharded buffers.
// Implementations must be thread-safe.
class Array : public llvm::RTTIExtends<Array, llvm::RTTIRoot> {
 public:
  Array() = default;

  // Not copyable or movable.
  Array(const Array&) = delete;
  Array(Array&&) = delete;
  Array& operator=(const Array&) = delete;
  Array& operator=(Array&&) = delete;

  virtual Client* client() const = 0;

  virtual DType dtype() const = 0;
  virtual const Shape& shape() const = 0;
  virtual const Sharding& sharding() const = 0;
  virtual std::shared_ptr<const Sharding> shared_ptr_sharding() const = 0;

  // Breaks an array up into per-device arrays. This is the elimination
  // counterpart of `Client::AssembleArrayFromSingleDeviceArrays()`.
  virtual StatusOr<std::vector<std::unique_ptr<Array>>>
  DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) = 0;

  // Fetches the array to host and stores it as unreplicated, unsharded data.
  //
  // DType whose sizes are unknown are unsupported.
  //
  // It may fail if sharding has insufficient information to
  // unreplicated/unshard the data (e.g., `OpaqueSharding`), or the sharding
  // contains an unaddressable device from the local runtime.
  //
  // If byte_strides is omitted, it defaults to a dense layout with dimensions
  // in major-to-minor order. The runtime may return `UNIMPLEMENTED` if
  // byte_strides does not equate to a reordering of the dimensions.
  //
  // `data` must remain valid until the returned future becomes ready. It will
  // contain a valid data only if the returned future has an OK. Otherwise, its
  // content is undefined.
  ABSL_MUST_USE_RESULT
  virtual Future<Status> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) = 0;

  // Copies the array with a new sharding, creating a new array.
  //
  // Resharding falls into one of the three cases:
  //
  // * Metadata-only resharding: Use a new sharding for the array that expects
  //   the same physical layout of underlying buffers on the same devices.
  // * 1-to-1 buffer copy: Copy individual buffers to different devices without
  //   altering their physical layout.
  // * M-to-N buffer resharding: Shuffle the buffer data across the boundary of
  //   the buffers, changing their physical layout.
  //
  // Implementations may return `UNIMPLEMENTED` if they do not know how to copy
  // or reshuffle the data to match the new sharding.
  //
  // It may fail if the buffer data would be sent from/to an unaddressable
  // device.
  virtual StatusOr<std::unique_ptr<Array>> Reshard(
      std::shared_ptr<const Sharding> new_sharding,
      ArrayCopySemantics semantics) = 0;

  // Returns a future that becomes ready when the buffer is computed or has an
  // error.
  virtual Future<Status> GetReadyFuture() const = 0;

  // Deletes the array from the devices. The operation may be asynchronous. The
  // returned future will have the result of the deletion on the devices.
  // Implementations that do not track the completion of the deletion operation
  // may make the future immediately ready with an OK status.
  virtual Future<Status> Delete() = 0;

  // Returns whether the array has been enqueued for deletion from the devices.
  virtual bool IsDeleted() const = 0;

  virtual std::string DebugString() const = 0;

  static char ID;  // NOLINT
};

// Convenience function to create a list of pointer Arrays from a list of
// unique_ptr Arrays.
std::vector<Array*> MakeArrayPointerList(
    absl::Span<const std::unique_ptr<Array>> arrays);

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_ARRAY_H_
