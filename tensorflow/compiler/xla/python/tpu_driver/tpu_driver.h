// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_TPU_DRIVER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_TPU_DRIVER_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/tpu_driver/platform/external/compat.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"

// This API is EXPERIMENTAL and under active development. It is subject to
// change without notice.

namespace tpu_driver {

int64_t ComputeBytesFromShape(const xla::ShapeProto& shape);

// Represents the deferred completion of a scheduled operation.
//
// Events may be blocked on, or used as `wait_for` arguments to enforce
// inter-operation dependencies.
class Event {
 public:
  virtual ~Event() {}

  // Blocks until the event completes and returns the result status.
  virtual xla::Status Await() = 0;
  // Returns an empty result if the wait times out.
  virtual std::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) = 0;

  // If the event is already done, the callback is called immediately.
  virtual void AddCallback(std::function<void(xla::Status)> callback) = 0;
};

// Represents a device memory allocation.
class BufferHandle {
 public:
  virtual ~BufferHandle() {}

  // This event completes after the device memory is actually allocated.
  //
  // Methods that take a buffer handle, such as ExecuteProgram and Transfer*,
  // automatically add this event as a dependency.
  virtual std::shared_ptr<Event> OnReady() = 0;

  virtual int64_t size_in_bytes() = 0;
  virtual std::optional<xla::ShapeProto> shape() = 0;
};

// Represents a compiled program on the host.
class CompiledProgramHandle {
 public:
  virtual ~CompiledProgramHandle() {}

  // This Event completes after the program is actually compiled on the host.
  //
  // Methods that take a compiled program handle, including LoadProgram,
  // automatically add this event as a dependency.
  virtual std::shared_ptr<Event> OnReady() = 0;

  virtual int64_t size_in_bytes() {
    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

  // Returns the shape of the compiled program. Blocks until compile completes.
  virtual xla::Status program_shape(xla::ProgramShapeProto* program_shape) = 0;
};

// Represents a program loaded on the device.
class LoadedProgramHandle {
 public:
  virtual ~LoadedProgramHandle() {}

  // This Event completes after the program is actually loaded on the device.
  //
  // Methods that take a loaded program handle, including ExecuteProgram and
  // UnloadProgram, automatically add this event as a dependency.
  virtual std::shared_ptr<Event> OnReady() = 0;

  virtual int64_t size_in_bytes() {
    LOG(FATAL) << "Unimplemented.";
    return 0;
  }
};

// A TpuLinearizer manages the linearization and delinearization of user buffers
// in the TPU driver. This interface is not yet implemented.
class TpuLinearizer {
 public:
  virtual ~TpuLinearizer() {}

  int64_t ComputeBytesFromShape(const xla::ShapeProto& shape) {
    return ::tpu_driver::ComputeBytesFromShape(shape);
  }
  virtual int64_t ComputeLinearizedBytesFromShape(
      const xla::ShapeProto& shape) = 0;

  virtual xla::Status LinearizeShape(void* dst, const void* src,
                                     const xla::ShapeProto& shape) = 0;
  virtual xla::Status DelinearizeShape(void* dst, const void* src,
                                       const xla::ShapeProto& shape) = 0;
};

// A TpuDriver manages a set of operations scheduled to run on a TPU system.
//
// By default, two independently scheduled operations may execute in any order.
// Ordering can be imposed in one of two ways:
//
// 1. Users can specify event dependencies via the `wait_for` argument.
// 2. Operations using buffer or program handles implicitly wait for the handles
//    to become ready before executing.
//
// For returned handle objects, the user is responsible for calling the release
// methods (Deallocate, UnloadProgram, etc.) that consume the given unique_ptr
// arguments and free up device resources. For returned event objects, there is
// no release method; the user can let them go out of scope naturally. As soon
// as those methods accepting plain-pointer arguments return, the user can let
// the corresponding smart-pointer objects be released or go out of scope,
// regardless of whether the scheduled device operations have started execution.
class TpuDriver {
 public:
  virtual ~TpuDriver() {}

  virtual void QuerySystemInfo(SystemInfo* system_info) = 0;
  // Synchronous. Reset the state of the TPU driver. After Reset(), this TPU
  // driver object is no longer usable. Users must destroy this object and
  // create a new one.
  //
  // All running programs will be terminated and all allocations reset. All
  // events and buffer handles created prior to Reset() will be invalid, and any
  // use will result in undefined behavior.
  virtual xla::Status Reset() = 0;

  virtual std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) = 0;

  // Allocate a buffer representing a tuple of `children` buffers.
  //
  // The returned tuple buffer handle does not manage the memory of `children`:
  // all `children` buffer handles must outlive the last usage of this tuple
  // buffer handle. One way to guarantee that is to deallocate the tuple buffer
  // handle before deallocating any buffer handle in `children`.
  //
  // All `children` buffers must exist in the same `core_id` and `region`.
  // If `children` is empty, a zero-sized tuple will be allocated in `region`.
  virtual std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) = 0;

  /* For buffers declared with an xla::ShapeProto rather than a raw size,
   * `src` must be laid out in consecutive row-major format for ingestion, and
   * each element must take up the number of bytes specified by the type.
   *
   * For example, for a [3,3,3] tensor with a Float32 type, the memory layout
   * would be as follows:
   *
   * [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], ..., [0,2,2], [1,0,0], ...
   * [1,2,2], [2,0,0], ..., [2,2,2],
   *
   * and the entire buffer will be 108 bytes (27 elements x 4 bytes).
   *
   * See
   * https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
   * for a more detailed description.
   *
   * `TransferFromDevice` will write out the shape back in this order as well.
   */
  virtual std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) = 0;

  virtual std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) = 0;

  virtual std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) = 0;

  virtual std::unique_ptr<TpuLinearizer> GetLinearizer() { return nullptr; }
};

class TpuDriverRegistry {
 public:
  static xla::StatusOr<std::unique_ptr<TpuDriver>> Open(
      const TpuDriverConfig& config);
  static int RegisterDriver(
      const std::string& prefix,
      const std::function<xla::StatusOr<std::unique_ptr<TpuDriver>>(
          const TpuDriverConfig&)>& creator);
};

#define REGISTER_TPU_DRIVER(prefix, fn) \
  REGISTER_TPU_DRIVER_HELPER(__COUNTER__, prefix, fn)
#define REGISTER_TPU_DRIVER_HELPER(ctr, prefix, fn)   \
  static int register_tpu_driver_count_unused_##ctr = \
      ::tpu_driver::TpuDriverRegistry::RegisterDriver(prefix, fn);

}  // namespace tpu_driver

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_TPU_DRIVER_H_
