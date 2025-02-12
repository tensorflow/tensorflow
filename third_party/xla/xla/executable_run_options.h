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

#ifndef XLA_EXECUTABLE_RUN_OPTIONS_H_
#define XLA_EXECUTABLE_RUN_OPTIONS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

// These classes are forward declared so that ExecutableRunOptions can be linked
// into an XLA-compiled binary without having to link all of the pointed-to
// objects (e.g., for an ahead-of-time compiled CPU binary, the gpu tools don't
// need to be linked).
namespace stream_executor {
class Stream;
class Event;
class Platform;
class DeviceMemoryAllocator;
class DeviceMemoryBase;
}  // namespace stream_executor

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace tsl {
template <typename T>
class AsyncValueRef;
}  // namespace tsl

namespace xla {

class DeviceAssignment;
class ExecutionProfile;
class Shape;

namespace cpu {
class CpuExecutableRunOptions;
}  // namespace cpu

namespace gpu {
class GpuExecutableRunOptions;
}  // namespace gpu

namespace ffi {
class ExecutionContext;
}  // namespace ffi

// A unique identifier for a particular "logical execution" of an XLA model.
//
// A logical execution might encompass multiple executions of one or more
// HloModules.  Runs that are part of the same logical execution can
// communicate via collective ops (e.g. kAllToAll), whereas runs that are part
// of different logical executions are isolated.
class RunId {
 public:
  // Creates a new, unique RunId.
  RunId();
  explicit RunId(int64_t value) : data_(value) {}

  RunId(const RunId&) = default;
  RunId& operator=(const RunId&) = default;
  friend bool operator==(const RunId& a, const RunId& b);
  std::string ToString() const;
  int64_t ToInt() const;

  template <typename H>
  friend H AbslHashValue(H h, const RunId& id) {
    return H::combine(std::move(h), id.data_);
  }

 private:
  int64_t data_;
};

// Callback used by the GPU backend only. This is an "one-sided" version of
// ThenDoHostCallback that enqueues a callback onto a stream. The difference
// with ThenDoHostCallback is that the device does not block waiting for the
// callback to complete; instead the callback is scheduled by the runtime.
// This functionality must be provided by the caller, and hence is provided in
// callback form.
using ThenExecuteFunction =
    std::function<void(stream_executor::Stream*, std::function<void()>)>;

// Callback for sending device buffer to a channel. Returned event will be
// recorded on a `stream` once the send operation is completed and data was
// copied from the `src` memory. `frontend_attrs` contains frontend specific
// attributes for the send.
using SendDeviceMemoryFunction = std::function<
    absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<stream_executor::Event>>>(
        int64_t channel_id, stream_executor::Stream* stream, const Shape& shape,
        const stream_executor::DeviceMemoryBase& src,
        const absl::flat_hash_map<std::string, std::string>& frontend_attrs)>;

// Callback for receiving device buffer from a channel. Returned event will be
// recorded on a `stream` once the recv operation is completed and data was
// copied into the `dst` memory. `frontend_attrs` contains frontend specific
// attributes for the receive.
using RecvDeviceMemoryFunction = std::function<
    absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<stream_executor::Event>>>(
        int64_t channel_id, stream_executor::Stream* stream, const Shape& shape,
        stream_executor::DeviceMemoryBase* dst,
        const absl::flat_hash_map<std::string, std::string>& frontend_attrs)>;

// Class containing options for running a LocalExecutable.
class ExecutableRunOptions {
 public:
  // Specifies the allocator to use during execution.
  ExecutableRunOptions& set_allocator(
      stream_executor::DeviceMemoryAllocator* allocator);
  stream_executor::DeviceMemoryAllocator* allocator() const;

  // If set, this is the device to run the computation on. Valid device_ordinal
  // values are: 0 to # of devices - 1. These are the logical device ordinals,
  // since multiple logical devices could reside on the same physical device,
  // e.g., virtual GPUs. If there is only one logical device on a physical
  // device, then these values are identical to the device ordinal values used
  // by StreamExecutor. The device must be of the same type as the executable
  // was compiled for. A value of -1 indicates this option has not been set.
  ExecutableRunOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this is the physical device to run the computation on. These values
  // are identical to the device ordinal values used by StreamExecutor. The
  // device must be of the same type as the executable was compiled for. A value
  // of -1 indicates this option has not been set, in which case the physical
  // device ordinal is the same as the logical device ordinal.
  ExecutableRunOptions& set_physical_device_ordinal(
      int physical_device_ordinal);
  int physical_device_ordinal() const;

  // If set, this is the stream to run the computation on. The platform of the
  // stream must match the platform the executable was built for.  A value of
  // nullptr indicates the option has not been set.
  ExecutableRunOptions& set_stream(stream_executor::Stream* stream);
  stream_executor::Stream* stream() const;

  // If set, this is the stream to perform host to device transfers on (e.g. any
  // pre-computation transfers). The platform of the stream must match the
  // platform the executable was built for. A value of nullptr indicates the
  // option has not been set.
  ExecutableRunOptions& set_host_to_device_stream(
      stream_executor::Stream* stream);
  stream_executor::Stream* host_to_device_stream() const;

  // If set, this is the stream to perform device to host transfers on.
  // The platform of the stream must match the platform the executable was
  // built for. A value of nullptr indicates the option has not been set.
  ExecutableRunOptions& set_device_to_host_stream(
      stream_executor::Stream* stream);
  stream_executor::Stream* device_to_host_stream() const;

  // Sets the thread pool device on which to run Eigen subcomputations.
  //
  // This field must be set for XLA:CPU models that call Eigen routines, but may
  // be null otherwise.  Routines that use this field should always CHECK (or
  // TF_RET_CHECK) that it's not null before dereferencing it, so that users get
  // a clean crash rather than a segfault.
  //
  // Does not take ownership.
  ExecutableRunOptions& set_intra_op_thread_pool(
      const Eigen::ThreadPoolDevice* intra_op_thread_pool);
  const Eigen::ThreadPoolDevice* intra_op_thread_pool() const;

  // If set, profiling information is written to 'profile'.
  ExecutionProfile* execution_profile() const;
  ExecutableRunOptions& set_execution_profile(ExecutionProfile* profile);

  ExecutableRunOptions& set_device_assignment(
      const DeviceAssignment* device_assignment);
  const DeviceAssignment* device_assignment() const;

  ExecutableRunOptions& set_rng_seed(int rng_seed);
  int rng_seed() const;

  ExecutableRunOptions& set_launch_id(int32_t launch_id) {
    launch_id_ = launch_id;
    return *this;
  }

  int32_t launch_id() const { return launch_id_; }

  ExecutableRunOptions& set_run_id(RunId id);
  RunId run_id() const;

  // See documentation on ThenExecuteFunction.
  ExecutableRunOptions& set_then_execute_function(ThenExecuteFunction* f) {
    then_execute_function_ = f;
    return *this;
  }
  ThenExecuteFunction* then_execute_function() const {
    return then_execute_function_;
  }

  // See documentation on SendDeviceMemoryFunction.
  ExecutableRunOptions& set_send_device_memory_function(
      SendDeviceMemoryFunction* f) {
    send_device_memory_function_ = f;
    return *this;
  }
  SendDeviceMemoryFunction* send_device_memory_function() const {
    return send_device_memory_function_;
  }

  // See documentation on RecvDeviceMemoryFunction.
  ExecutableRunOptions& set_recv_device_memory_function(
      RecvDeviceMemoryFunction* f) {
    recv_device_memory_function_ = f;
    return *this;
  }
  RecvDeviceMemoryFunction* recv_device_memory_function() const {
    return recv_device_memory_function_;
  }

  // CPU-backend specific options. These are kept out-of-line to avoid bloating
  // the size of this dependency for CPU-only AOT builds.
  ExecutableRunOptions& set_cpu_executable_run_options(
      const cpu::CpuExecutableRunOptions* cpu_executable_run_options);
  const cpu::CpuExecutableRunOptions* cpu_executable_run_options() const;

  // GPU-backend specific options. These are kept out-of-line to avoid bloating
  // the size of this dependency for CPU-only AOT builds.
  ExecutableRunOptions& set_gpu_executable_run_options(
      const gpu::GpuExecutableRunOptions* gpu_executable_run_options);
  const gpu::GpuExecutableRunOptions* gpu_executable_run_options() const;

  // XLA FFI specific execution context that allows to pass auxiliary data to
  // FFI handlers. It's a caller responsibility to ensure that the XLA FFI
  // execution context stays alive while the executable is running.
  ExecutableRunOptions& set_ffi_execution_context(
      const ffi::ExecutionContext* ffi_execution_context);
  const ffi::ExecutionContext* ffi_execution_context() const;

  // This indicates how many local devices are used by the execution.
  // Valid values are any value greater than 0.
  // 0 means unset.
  ExecutableRunOptions& set_local_device_count(int local_device_count);
  int local_device_count() const;

 private:
  stream_executor::DeviceMemoryAllocator* allocator_ = nullptr;
  int device_ordinal_ = -1;
  int local_device_count_ = 0;
  int physical_device_ordinal_ = -1;
  const DeviceAssignment* device_assignment_ = nullptr;
  stream_executor::Stream* stream_ = nullptr;
  const Eigen::ThreadPoolDevice* intra_op_thread_pool_ = nullptr;
  ExecutionProfile* execution_profile_ = nullptr;
  int rng_seed_ = 0;
  int32_t launch_id_ = 0;
  stream_executor::Stream* device_to_host_stream_ = nullptr;
  stream_executor::Stream* host_to_device_stream_ = nullptr;
  ThenExecuteFunction* then_execute_function_ = nullptr;
  SendDeviceMemoryFunction* send_device_memory_function_ = nullptr;
  RecvDeviceMemoryFunction* recv_device_memory_function_ = nullptr;
  RunId run_id_;
  const cpu::CpuExecutableRunOptions* cpu_executable_run_options_ = nullptr;
  const gpu::GpuExecutableRunOptions* gpu_executable_run_options_ = nullptr;
  const ffi::ExecutionContext* ffi_execution_context_ = nullptr;
};

}  // namespace xla

#endif  // XLA_EXECUTABLE_RUN_OPTIONS_H_
