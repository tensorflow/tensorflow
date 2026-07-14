/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PARAMS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PARAMS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// Parameters capturing all the details required for collective execution of
// XLA executables (multiple partitions and replicas).
struct CollectiveParams {
  // A mapping from local device ordinals to global device IDs.
  using GlobalDeviceIdMap = GpuExecutableRunOptions::DeviceIdMap;

  // Creates collective execution parameters from the run options for the given
  // local device. Returns an error if run options are misconfigured (i.e.
  // missing a global device mapping for a local device ordinal).
  static absl::StatusOr<CollectiveParams> Create(
      const ServiceExecutableRunOptions& run_options,
      absl::Span<se::Stream* const> async_streams,
      LocalDeviceId local_device_id,
      std::optional<std::string> implementation_name = std::nullopt,
      int64_t collective_max_nchannels = 0, int64_t p2p_max_nchannels = 0,
      bool collective_use_minimal_resource = false);

  GpuCollectives* collectives;
  se::StreamExecutor* executor;

  // XLA execution run id allows us to distinguish collective operations
  // from different concurrent executions and avoid deadlocks.
  RunId run_id;

  // Streams for asynchronous collective communications.
  absl::InlinedVector<se::Stream*, 4> async_streams;

  LocalDeviceId local_device_id;
  GlobalDeviceId global_device_id;

  const DeviceAssignment* device_assn;
  const GlobalDeviceIdMap* global_device_id_map;
  const CliqueIdCallback* clique_id_callback;
  const absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations;

  int64_t collective_max_nchannels;
  int64_t p2p_max_nchannels;

  int local_device_count = 0;
  bool need_barrier = false;
  bool collective_use_minimal_resource;

 private:
  CollectiveParams(
      GpuCollectives* collectives, se::StreamExecutor* executor, RunId run_id,
      absl::Span<se::Stream* const> async_streams,
      LocalDeviceId local_device_id, GlobalDeviceId global_device_id,
      const DeviceAssignment* device_assn,
      const GlobalDeviceIdMap* global_device_id_map,
      const CliqueIdCallback* clique_id_callback,
      const absl::flat_hash_map<GlobalDeviceId, IncarnationId>* incarnations,
      int64_t collective_max_nchannels, int64_t p2p_max_nchannels,
      int local_device_count, bool collective_use_minimal_resource);
};

// The type of a kernel argument used by a collective kernel.
enum class KernelArgType : uint8_t {
  // Pointer to an input buffer.
  kInputBuffer,
  // Pointer to an output buffer.
  kOutputBuffer,
  // Scalar. The rank of the current device.
  kRuntimeRank,
  // The number of times the collective kernel has been invoked. For example
  // inside a while loop.
  kInvocationCount,
  // A scratch buffer.
  kScratchBuffer,
};

// Specifies how symmetric memory allocation and pointer exchange are handled
// for a collective kernel buffer.
enum class SymmetricMemoryType {
  // The runtime performs no symmetric memory registration or pointer exchange.
  kNone = 0,
  // The runtime registers the buffer for direct load/store fabric access across
  // peers (e.g., via ncclCommRegister / window registration in NCCL or ROCm
  // equivalents).
  kLoadStoreAccessible = 1,
  // Pointers are exchanged across devices via host-side rendezvous during
  // initialization.
  // NOTE: Because this exchange occurs once during initialization, this type
  // CANNOT be used on standard HLO buffers (which may be aliased or reused for
  // other operations, making the initial pointer exchange invalid). It must
  // ONLY be used for buffers allocated by the thunk outside of XLA, such as
  // scratch buffers.
  kXlaRendezvous = 2
};

// Specification for an input/output buffer used by a collective kernel.
struct IoBufferSpec {
  // Whether the buffer requires allocation for multicast collectives.
  bool requires_multimem;
  // How symmetric memory is allocated and managed for this buffer.
  SymmetricMemoryType symmetric_memory_type;
};

// Specification for a scratch buffer used by a collective kernel.
struct ScratchBufferSpec {
  // The size of the scratch buffer in bytes.
  int64_t size_bytes;
  // Whether the buffer requires registration for multicast collectives.
  bool requires_multimem;
  // How symmetric memory is allocated and managed for this buffer.
  SymmetricMemoryType symmetric_memory_type;
  // Whether the buffer should be memzeroed before use.
  bool should_memzero;
  // Whether the buffer should be double buffered.
  // If true, the buffer will be allocated as size_bytes * 2 and
  // interleaved to avoid synchronization between successive kernel invocations.
  bool should_double_buffer;
};

// Specification for a kernel argument used by a collective kernel.
struct KernelArgDescriptor {
  KernelArgType type;
  // The index of the buffer in the collective kernel spec. Only used for
  // buffer arguments. For example, if there are two operand buffers, two result
  // buffers, and 4 scratch buffers, the valid indices for operand/result
  // buffers are 0, 1 and for scratch buffers are 0, 1, 2, 3.
  // For scalar arguments, the index is set to std::nullopt.
  std::optional<int32_t> index = std::nullopt;
};

// This structure contains the information required to configure and launch a
// custom collective kernel.
struct CollectiveKernelSpec {
  // Specs for input operand buffers.
  std::vector<IoBufferSpec> input_buffer_specs;
  // Specs for output result buffers.
  std::vector<IoBufferSpec> output_buffer_specs;
  // Specs for scratch buffers these are allocated by the thunk.
  std::vector<ScratchBufferSpec> scratch_buffers;
  // Argument descriptors that determine how the kernel is invoked.
  std::vector<KernelArgDescriptor> argument_descriptors;
  // Each time ExecuteOnStream is called, the invocation count is incremented by
  // this amount. For one-shot collectives, this is 1. For two-shot collectives,
  // this is 2 and so on.
  uint32_t sync_count_increment = 1;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PARAMS_H_
