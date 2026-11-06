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

#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/mock_gpu_collectives.h"
#include "xla/backends/gpu/collectives/mock_gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/future.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/ptx_compile_options_from_debug_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/assemble_compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::StrictMock;

static constexpr absl::string_view kProfileName = "test_kernel_profiler";
static constexpr absl::string_view kKernelName = "five_argument_kernel";
static constexpr int64_t kNumElements = 128;

TEST(CollectiveKernelThunkTest, ToStringIncludesPdlOnlyWhenEnabled) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = kProfileName;
  for (bool use_pdl : {false, true}) {
    CollectiveKernelThunk thunk(
        thunk_info, CollectiveConfig{}, CollectiveKernelSpec{},
        /*buffers=*/{}, /*is_collective_kernel_enabled=*/true, kKernelName,
        LaunchDimensions(se::BlockDim(3, 2, 1), se::ThreadDim(4, 5, 6)),
        /*devices_in_process=*/1, /*shmem_bytes=*/42,
        /*cubin=*/std::nullopt, use_pdl);
    EXPECT_EQ(thunk.ToString(0),
              std::string("kernel=five_argument_kernel, "
                          "profile_annotation=test_kernel_profiler, "
                          "launch dimensions=blocks: {3, 2, 1}, "
                          "threads/block: {4, 5, 6}, "
                          "cluster_dim=nullopt, shmem_bytes=42") +
                  (use_pdl ? ", use_pdl=true" : ""));
  }
}

// Test kernel was compiled using following CUDA source:
// __global__ void five_argument_kernel(int64_t* input_buffer,         // 1
//                                      int64_t* output_buffer,        // 2
//                                      int64_t rank,                  // 3
//                                      int32_t** signal_buffers,      // 4
//                                      int64_t* remote_buffers        // 5
// ) {
//   (void)remote_buffers;
//   // Mirrors EmitDeviceInvocationCount: the invocation count is derived from
//   // this block's barrier slot SignalBuffers[rank][blockIdx.x * world_size +
//   // rank], without a thread barrier or a store. The test launches a single
//   // block, so the slot index reduces to `rank`.
//   int32_t* local_signal_buffer = signal_buffers[rank];
//   int32_t signal_count =
//       *(volatile int32_t*)&local_signal_buffer[rank] + 1;
//   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   for (int i = idx; i < kNumElements; i += gridDim.x * blockDim.x) {
//     if (i < kNumElements) {
//       output_buffer[i] = input_buffer[i] + signal_count;
//     }
//   }
// }
static constexpr absl::string_view kKernelSource = R"(
  .version 8.7
  .target sm_90
  .address_size 64

  .visible .entry five_argument_kernel(
  .param .u64 .ptr .align 1 input_buffer,
  .param .u64 .ptr .align 1 output_buffer,
  .param .u64 rank,
  .param .u64 .ptr .align 1 signal_buffers,
  .param .u64 .ptr .align 1 remote_buffers
  )
  {
  .reg .pred %p<3>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<14>;

  ld.param.b64 %rd4, [input_buffer];
  cvta.to.global.u64 %rd1, %rd4;
  ld.param.b64 %rd5, [output_buffer];
  cvta.to.global.u64 %rd2, %rd5;
  ld.param.u32 %r7, [rank];
  ld.param.b64 %rd11, [signal_buffers];
  cvta.to.global.u64 %rd11, %rd11;
  mul.wide.u32 %rd12, %r7, 8;
  add.s64 %rd11, %rd11, %rd12;
  ld.global.u64 %rd13, [%rd11];
  cvta.to.global.u64 %rd13, %rd13;
  mul.wide.u32 %rd12, %r7, 4;
  add.s64 %rd13, %rd13, %rd12;
  ld.volatile.global.s32 %r7, [%rd13];
  add.s32 %r7, %r7, 1;
  cvt.s64.s32 %rd3, %r7;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r1, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r6, %r3, %r1, %r4;
  setp.gt.s32 %p1, %r6, 127;
  @%p1 bra $L__BB0_3;
  //
  mov.u32 %r5, %nctaid.x;
  mul.lo.s32 %r2, %r5, %r1;
  $L__BB0_2: //
  mul.wide.s32 %rd6, %r6, 8;
  add.s64 %rd7, %rd1, %rd6;
  ld.global.b64 %rd8, [%rd7];
  add.s64 %rd9, %rd8, %rd3;
  add.s64 %rd10, %rd2, %rd6;
  st.global.b64 [%rd10], %rd9;
  add.s32 %r6, %r6, %r2;
  setp.lt.s32 %p2, %r6, 128;
  @%p2 bra $L__BB0_2;
  $L__BB0_3:
  ret;
  }
  )";

// Test kernel was compiled using following CUDA source:
// extern "C" __global__ void five_argument_kernel(
//     int64_t* input_buffer,         // 1
//     int64_t** output_buffers,      // 2
//     int32_t rank,                  // 3
//     int32_t** signal_buffers,      // 4
//     int64_t* remote_buffers        // 5
// ) {
//   (void)remote_buffers;
//   int64_t* output_buffer = output_buffers[rank];
//   int32_t* local_signal_buffer = signal_buffers[rank];
//   int32_t signal_count =
//       *(volatile int32_t*)&local_signal_buffer[rank] + 1;
//   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   for (int i = idx; i < kNumElements; i += gridDim.x * blockDim.x) {
//     if (i < kNumElements) {
//       output_buffer[i] = input_buffer[i] + signal_count;
//     }
//   }
// }
static constexpr absl::string_view kSymmetricOutputKernelSource = R"(
  .version 8.7
  .target sm_90
  .address_size 64

  .visible .entry five_argument_kernel(
    .param .u64 .ptr .align 1 five_argument_kernel_param_0,
    .param .u64 .ptr .align 1 five_argument_kernel_param_1,
    .param .b32 five_argument_kernel_param_2,
    .param .u64 .ptr .align 1 five_argument_kernel_param_3,
    .param .u64 .ptr .align 1 five_argument_kernel_param_4
  )
  {
    .reg .pred %p<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<20>;

    ld.param::entry.b64 %rd5, [five_argument_kernel_param_1];
    cvta.to.global.u64 %rd6, %rd5;
    ld.param::entry.b64 %rd7, [five_argument_kernel_param_3];
    cvta.to.global.u64 %rd8, %rd7;
    ld.param::entry.b32 %r4, [five_argument_kernel_param_2];
    mul.wide.s32 %rd9, %r4, 8;
    add.s64 %rd10, %rd6, %rd9;
    ld.global.b64 %rd2, [%rd10];
    add.s64 %rd11, %rd8, %rd9;
    ld.global.b64 %rd12, [%rd11];
    mul.wide.s32 %rd13, %r4, 4;
    add.s64 %rd14, %rd12, %rd13;
    ld.volatile.b32 %r1, [%rd14];
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r9, %r5, %r2, %r6;
    setp.gt.s32 %p1, %r9, 127;
    @%p1 bra $L__BB0_3;
    ld.param::entry.b64 %rd4, [five_argument_kernel_param_0];
    cvta.to.global.u64 %rd1, %rd4;
    add.s32 %r7, %r1, 1;
    cvt.s64.s32 %rd3, %r7;
    mov.u32 %r8, %nctaid.x;
    mul.lo.s32 %r3, %r8, %r2;
  $L__BB0_2:
    mul.wide.s32 %rd15, %r9, 8;
    add.s64 %rd16, %rd1, %rd15;
    ld.global.b64 %rd17, [%rd16];
    add.s64 %rd18, %rd17, %rd3;
    add.s64 %rd19, %rd2, %rd15;
    st.b64 [%rd19], %rd18;
    add.s32 %r9, %r9, %r3;
    setp.lt.s32 %p2, %r9, 128;
    @%p2 bra $L__BB0_2;
  $L__BB0_3:
    ret;
  }
  )";

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  absl::StatusOr<se::Platform*> platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName());
  CHECK_OK(platform.status());
  absl::StatusOr<se::StreamExecutor*> executor =
      (*platform)->ExecutorForDevice(device_ordinal);
  CHECK_OK(executor.status());
  return *executor;
}

struct CollectiveKernelThunkMetadata {
  BufferAllocation buffer_allocation;
  std::unique_ptr<CollectiveKernelThunk> thunk;
  int64_t total_buffer_size;
  int64_t input_data_size_bytes;
  int64_t aligned_input_size_bytes;
  int64_t num_devices;
  // If true, the PTX is not compiled into CUBIN and is passed to the thunk as
  // a string.
  bool use_ptx;
  std::vector<CollectiveThunk::Buffer> buffers;
  absl::string_view kernel_source = kKernelSource;
  std::unique_ptr<CollectiveMemoryCache> collective_memory_cache =
      std::make_unique<CollectiveMemoryCache>(/*num_devices=*/2);
  std::vector<se::DeviceAddressBase> device_buffers;
  // If set, the output slice lives in this second allocation (index 1), which
  // is backed by the same device buffer as `buffer_allocation` (index 0).
  std::optional<BufferAllocation> aliased_output_allocation;
};

CollectiveKernelSpec CreateCollectiveKernelSpec(
    int64_t num_elements, int64_t signal_size, int64_t remote_size,
    bool is_multimem_enabled,
    SymmetricMemoryType scratch_memory_type =
        SymmetricMemoryType::kLoadStoreAccessible,
    SymmetricMemoryType output_memory_type = SymmetricMemoryType::kNone,
    bool output_requires_multimem = false) {
  return {
      /*codegen_config=*/{
          /*copy_input_to_scratch=*/false,
          /*input_buffer_specs=*/
          {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
          /*output_buffer_specs=*/
          {{/*requires_multimem=*/output_requires_multimem,
            output_memory_type}},
          /*argument_descriptors=*/
          {{KernelArgType::kInputBuffer, 0},
           {KernelArgType::kOutputBuffer, 0},
           {KernelArgType::kRuntimeRank},
           {KernelArgType::kScratchBuffer, 0},
           {KernelArgType::kScratchBuffer, 1}},
      },
      /*scratch_buffers=*/
      {{signal_size, /*requires_multimem=*/false, scratch_memory_type,
        /*should_memzero=*/true,
        /*should_double_buffer=*/true},
       {remote_size,
        /*requires_multimem=*/is_multimem_enabled, scratch_memory_type,
        /*should_memzero=*/false,
        /*should_double_buffer=*/true}},
  };
}

CollectiveKernelThunkMetadata CreateCollectiveKernelThunk(
    int num_devices, int num_elements, bool is_multimem_enabled, bool use_ptx,
    SymmetricMemoryType scratch_memory_type =
        SymmetricMemoryType::kLoadStoreAccessible,
    SymmetricMemoryType output_memory_type = SymmetricMemoryType::kNone,
    std::optional<CollectiveKernelSpec> custom_spec = std::nullopt,
    bool alias_output_allocation = false) {
  const int64_t input_size_bytes = num_elements * sizeof(uint64_t);
  Shape input_shape = ShapeUtil::MakeShape(U64, {num_elements});
  ReplicaGroup replica_group;

  for (int device_number = 0; device_number < num_devices; ++device_number) {
    replica_group.add_replica_ids(device_number);
  }

  CollectiveConfig collective_config{
      /*operand_element_type=*/{PrimitiveType::F32},
      /*replica_groups=*/{replica_group},
      /*group_mode=*/
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA,
      /*use_symmetric_buffer=*/false};
  const int64_t aligned_input_size_bytes =
      xla::RoundUpTo<uint64_t>(input_size_bytes, kXlaAllocatedBufferAlignBytes);
  // 2x because we have two buffers, one for input and one for output so we
  // can test output independently of input.
  const int64_t total_buffer_size = aligned_input_size_bytes * 2;
  CollectiveKernelThunkMetadata result{
      BufferAllocation(/*index=*/0, /*size=*/total_buffer_size, /*color=*/0)};
  const BufferAllocation* output_allocation = &result.buffer_allocation;
  if (alias_output_allocation) {
    result.aliased_output_allocation.emplace(/*index=*/1,
                                             /*size=*/total_buffer_size,
                                             /*color=*/0);
    output_allocation = &*result.aliased_output_allocation;
  }
  BufferAllocation::Slice input_slice(&result.buffer_allocation, /*offset=*/0,
                                      /*size=*/aligned_input_size_bytes);
  BufferAllocation::Slice output_slice(
      output_allocation, aligned_input_size_bytes, aligned_input_size_bytes);
  result.buffers = {{/*element_count=*/num_elements,
                     /*source_buffer=*/{input_slice, input_shape},
                     /*destination_buffer=*/{output_slice, input_shape},
                     /*source_memory_space=*/0,
                     /*destination_memory_space=*/0}};
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = kProfileName;
  const LaunchDimensions launch_dimensions(
      /*block_x_count=*/1, /*thread_x_count_per_block=*/kNumElements);
  const int64_t signal_size = 1024;
  const int64_t remote_size = aligned_input_size_bytes;
  CollectiveKernelSpec spec =
      custom_spec.has_value()
          ? *std::move(custom_spec)
          : CreateCollectiveKernelSpec(num_elements, signal_size, remote_size,
                                       is_multimem_enabled, scratch_memory_type,
                                       output_memory_type);
  result.thunk = std::make_unique<CollectiveKernelThunk>(
      std::move(thunk_info), collective_config, std::move(spec), result.buffers,
      /*is_collective_kernel_enabled=*/true,
      /*kernel_name=*/kKernelName,
      /*launch_dimensions=*/launch_dimensions,
      /*devices_in_process=*/num_devices,
      /*shmem_bytes=*/0);
  result.total_buffer_size = total_buffer_size;
  result.num_devices = num_devices;
  result.aligned_input_size_bytes = aligned_input_size_bytes;
  result.input_data_size_bytes = input_size_bytes;
  result.use_ptx = use_ptx;
  result.device_buffers.resize(num_devices);
  if (output_memory_type != SymmetricMemoryType::kNone) {
    result.kernel_source = kSymmetricOutputKernelSource;
  }
  return result;
}

// Compiles a PTX string to a CUBIN using the NVPTXCompiler.
//
// Args:
//   ptx_string: The PTX code to compile.
//   device_description: The description of the target GPU device.
//   debug_options: The debug options for configuring the compilation.
//
// Returns:
//   A StatusOr containing the compiled CUBIN as a vector of bytes.
absl::StatusOr<std::vector<uint8_t>> CompilePtxToCubin(
    const absl::string_view ptx_string,
    const se::DeviceDescription& device_description,
    const DebugOptions& debug_options) {
  se::cuda::CompilationProviderOptions options =
      se::cuda::CompilationProviderOptions::FromDebugOptions(debug_options);
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<se::cuda::CompilationProvider> compilation_provider,
      se::cuda::AssembleCompilationProvider(options));
  se::CudaComputeCapability cc =
      *device_description.gpu_compute_capability().cuda_compute_capability();
  se::cuda::CompilationOptions compilation_options =
      PtxCompileOptionsFromDebugOptions(debug_options);
  ABSL_ASSIGN_OR_RETURN(
      se::cuda::Assembly assembly,
      compilation_provider->Compile(cc, ptx_string, compilation_options));
  return std::move(assembly.cubin);
}

absl::StatusOr<se::DeviceAddressBase> RunCollectiveKernelThunk(
    CollectiveKernelThunkMetadata& metadata, se::StreamExecutor* executor,
    std::vector<uint64_t> input_data, bool emulate_multiprocess = false,
    GpuCollectives* collectives = nullptr) {
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0)),
      std::make_pair(LocalDeviceId(1), GlobalDeviceId(1))});
  if (collectives != nullptr) {
    gpu_options.set_collectives(collectives);
  }

  ABSL_ASSIGN_OR_RETURN(auto stream, executor->CreateStream());
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  DeviceAssignment device_assignment(/*replica_count=*/metadata.num_devices,
                                     /*computation_count=*/1);

  for (int i = 0; i < metadata.num_devices; ++i) {
    device_assignment(i, 0) = i;
  }

  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);

  ABSL_ASSIGN_OR_RETURN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  // We always allocate from collective memory space because we want to be able
  // to test multimem kernels. In XLA programs it's up to the compiler to assign
  // correct memory space to kernel parameters and results buffers.
  const int device_ordinal = executor->device_ordinal();
  if (metadata.device_buffers[device_ordinal].is_null()) {
    metadata.device_buffers[device_ordinal] = executor->AllocateArray<uint64_t>(
        metadata.total_buffer_size,
        /*memory_space=*/static_cast<int>(se::MemorySpace::kCollective));
  }
  std::vector<se::DeviceAddressBase> allocated_buffers = {
      metadata.device_buffers[device_ordinal]};
  if (metadata.aliased_output_allocation.has_value()) {
    // Allocation index 1 aliases the same device buffer as index 0.
    allocated_buffers.push_back(metadata.device_buffers[device_ordinal]);
  }

  se::DeviceAddressBase input_buffer =
      allocated_buffers[0].GetByteSlice(0, metadata.aligned_input_size_bytes);
  se::DeviceAddressBase output_buffer = allocated_buffers[0].GetByteSlice(
      metadata.aligned_input_size_bytes, metadata.aligned_input_size_bytes);
  BufferAllocations buffer_allocations(
      /*buffers=*/allocated_buffers,
      /*device_ordinal=*/executor->device_ordinal(),
      /*memory_allocator=*/nullptr);

  if (!input_data.empty()) {
    VLOG(3) << "Copying input data to the device";
    ABSL_RETURN_IF_ERROR(stream->Memcpy(&input_buffer, input_data.data(),
                                   metadata.input_data_size_bytes));
    ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  }

  CollectiveCliqueRequests clique_requests;

  ReplicaGroup all_replica_groups;
  for (int64_t i = 0; i < metadata.num_devices; ++i) {
    all_replica_groups.add_replica_ids(i);
  }
  // Request a clique that covers all devices (this test runs on 2 gpus).
  ABSL_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      xla::gpu::GetGpuCliqueKey(
          collective_params, {all_replica_groups},
          CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID));
  std::vector<GlobalDeviceId> all_device_groups;
  all_device_groups.reserve(metadata.num_devices);
  for (int i = 0; i < metadata.num_devices; ++i) {
    all_device_groups.push_back(GlobalDeviceId(i));
  }
  ABSL_RETURN_IF_ERROR(clique_requests.RequestClique(
      clique_key, /*device_groups=*/{all_device_groups}));
  CollectiveMemoryRequests memory_requests(buffer_allocations);
  CommandBufferRequests command_buffer_requests;
  Thunk::PrepareParams prepare_params{
      /*collective_params=*/&collective_params,
      /*collective_clique_requests=*/&clique_requests,
      /*collective_memory_requests=*/&memory_requests,
      /*executor=*/executor,
      /*buffer_allocations=*/&buffer_allocations,
      /*execution_scoped_state=*/nullptr,
      /*custom_options=*/nullptr,
      /*command_buffer_requests=*/&command_buffer_requests};

  ABSL_RETURN_IF_ERROR(metadata.thunk->Prepare(prepare_params));
  CollectiveCliques collective_cliques;
  ABSL_ASSIGN_OR_RETURN(collective_cliques, AcquireCollectiveCliques(
                                           collective_params, clique_requests));
  ABSL_ASSIGN_OR_RETURN(CollectiveMemory collective_memory,
                   AcquireCollectiveMemory(collective_params,
                                           collective_cliques, memory_requests,
                                           *metadata.collective_memory_cache));

  Thunk::InitializeParams initialize_params(&command_buffer_requests);
  initialize_params.executor = executor;
  initialize_params.src = {metadata.kernel_source};
  initialize_params.buffer_allocations = &buffer_allocations;
  initialize_params.stream = stream.get();
  initialize_params.collective_params = &collective_params;
  initialize_params.collective_cliques = &collective_cliques;
  initialize_params.collective_memory = &collective_memory;

  GpuExecutableRunOptions::DeviceIdMap global_device_id_map = {
      {LocalDeviceId(0), GlobalDeviceId(0)}};
  if (emulate_multiprocess) {
    initialize_params.collective_params->global_device_id_map =
        &global_device_id_map;
  }

  std::vector<uint8_t> cubin;
  if (!metadata.use_ptx) {
    ABSL_ASSIGN_OR_RETURN(cubin, CompilePtxToCubin(metadata.kernel_source,
                                              executor->GetDeviceDescription(),
                                              DebugOptions()));
    initialize_params.src.binary = cubin;
  }
  ABSL_RETURN_IF_ERROR(metadata.thunk->Initialize(initialize_params));

  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/&collective_memory);
  ABSL_RETURN_IF_ERROR(metadata.thunk->ExecuteOnStream(execute_params));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return output_buffer;
}

absl::StatusOr<std::vector<se::DeviceAddressBase>>
RunCollectiveKernelThunkOnDevices(CollectiveKernelThunkMetadata& metadata,
                                  bool emulate_multiprocess = false,
                                  const std::vector<uint64_t>& input_data = {},
                                  GpuCollectives* collectives = nullptr) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "device_threads",
                                      metadata.num_devices);

  std::vector<tsl::Future<se::DeviceAddressBase>> futures(metadata.num_devices);
  for (int d = 0; d < metadata.num_devices; ++d) {
    futures[d] = tsl::MakeFutureOn<se::DeviceAddressBase>(
        *thread_pool.AsExecutor(),
        [&metadata, d, emulate_multiprocess, &input_data, collectives] {
          return RunCollectiveKernelThunk(metadata, GetGpuExecutor(d),
                                          input_data, emulate_multiprocess,
                                          collectives);
        });
  }

  return JoinFutures<se::DeviceAddressBase>(futures).Await();
}

std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
MakeSequentialInputAndExpectedOutput(uint64_t base_value = 0,
                                     uint32_t expected_signal_value = 1) {
  std::vector<uint64_t> input_data(kNumElements);
  std::vector<uint64_t> expected_output_data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    input_data[i] = base_value + i;
    expected_output_data[i] = input_data[i] + expected_signal_value;
  }
  return {std::move(input_data), std::move(expected_output_data)};
}

void VerifyDeviceOutputs(absl::Span<const se::DeviceAddressBase> buffers,
                         int64_t input_data_size_bytes,
                         const std::vector<uint64_t>& expected) {
  for (size_t d = 0; d < buffers.size(); ++d) {
    se::StreamExecutor* executor = GetGpuExecutor(d);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                         executor->CreateStream());
    std::vector<uint64_t> output_data(expected.size());
    ASSERT_OK(
        stream->Memcpy(output_data.data(), buffers[d], input_data_size_bytes));
    ASSERT_OK(stream->BlockHostUntilDone());
    EXPECT_EQ(output_data, expected);
  }
}

class CollectiveKernelThunkParameterizedTest
    : public ::testing::TestWithParam<bool> {};

TEST_P(CollectiveKernelThunkParameterizedTest, ExecutesPtxKernel) {
  auto [input_data, expected_output_data] =
      MakeSequentialInputAndExpectedOutput();

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/1, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/GetParam());

  se::StreamExecutor* executor0 = GetGpuExecutor(0);
  ASSERT_OK_AND_ASSIGN(
      se::DeviceAddressBase result_buffer,
      RunCollectiveKernelThunk(metadata, executor0, input_data));
  VerifyDeviceOutputs({result_buffer}, metadata.input_data_size_bytes,
                      expected_output_data);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveKernelThunkParameterizedTest,
    CollectiveKernelThunkParameterizedTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& use_ptx) -> std::string {
      return use_ptx.param ? "UsingPtx" : "UsingCubin";
    });

TEST(CollectiveKernelThunkTest, MultimemSetupTest) {
  static constexpr int kDevicesCount = 2;

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/kDevicesCount, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/true, /*use_ptx=*/true);
  ASSERT_OK(RunCollectiveKernelThunkOnDevices(metadata));
}

TEST(CollectiveKernelThunkTest, MultiprocessTest) {
  static constexpr int kDevicesCount = 2;

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/kDevicesCount, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/true,
      /*scratch_memory_type=*/SymmetricMemoryType::kXlaRendezvous);
  EXPECT_THAT(RunCollectiveKernelThunkOnDevices(metadata,
                                                /*emulate_multiprocess=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CollectiveKernelThunkTest, BufferUses) {
  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/2, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/true);

  EXPECT_THAT(
      metadata.thunk->buffer_uses(),
      ElementsAre(
          BufferUse::Read(metadata.buffers[0].source_buffer.slice,
                          metadata.buffers[0].source_buffer.shape),
          BufferUse::Write(metadata.buffers[0].destination_buffer.slice,
                           metadata.buffers[0].destination_buffer.shape)));
}

bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

static CollectiveThunk::Buffer MakeNoOpBuffer(const BufferAllocation& alloc_src,
                                              const BufferAllocation& alloc_dst,
                                              int64_t length) {
  int64_t byte_length = sizeof(float) * length;
  ShapedSlice src_slice{BufferAllocation::Slice(&alloc_src, 0, byte_length),
                        ShapeUtil::MakeShape(F32, {length})};
  ShapedSlice dst_slice{BufferAllocation::Slice(&alloc_dst, 0, byte_length),
                        ShapeUtil::MakeShape(F32, {length})};
  CollectiveThunk::Buffer buffer{/*.element_count=*/length,
                                 /*.source_buffer=*/src_slice,
                                 /*.destination_buffer=*/dst_slice,
                                 /*.source_memory_space=*/0,
                                 /*.destination_memory_space=*/0};
  return buffer;
}

TEST(CollectiveKernelThunkTest, RecordCommandBufferCreateUpdate) {
  se::StreamExecutor* executor = GetGpuExecutor(0);
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  if (!executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper() &&
      !executor->GetDeviceDescription().gpu_compute_capability().IsRocm()) {
    GTEST_SKIP()
        << "Collective kernel requires Hopper or newer, or a ROCm device";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto trace_stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto async_stream, executor->CreateStream());

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(float) * kLength;

  se::DeviceAddress<float> src1 = executor->AllocateArray<float>(kLength, 0);
  se::DeviceAddress<float> dst1 = executor->AllocateArray<float>(kLength, 0);
  se::DeviceAddress<float> src2 = executor->AllocateArray<float>(kLength, 0);
  se::DeviceAddress<float> dst2 = executor->AllocateArray<float>(kLength, 0);

  BufferAllocation alloc_src(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, kByteLength, /*color=*/0);
  std::vector<CollectiveThunk::Buffer> buffers = {
      MakeNoOpBuffer(alloc_src, alloc_dst, kLength)};

  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);
  CollectiveConfig collective_config{
      /*operand_element_type=*/{F32},
      /*replica_groups=*/{replica_group},
      /*group_mode=*/
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA,
      /*use_symmetric_buffer=*/false};

  const LaunchDimensions launch_dimensions(1, kLength);

  int64_t num_elements = kLength;
  int64_t signal_size = 1;
  int64_t remote_size = 1;
  bool is_multimem_enabled = false;

  auto collective_kernel_thunk = std::make_unique<CollectiveKernelThunk>(
      Thunk::ThunkInfo(), collective_config,
      CreateCollectiveKernelSpec(num_elements, signal_size, remote_size,
                                 is_multimem_enabled),
      buffers, /*is_collective_kernel_enabled=*/true, std::string(kKernelName),
      launch_dimensions, /*devices_in_process=*/1);

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;

  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);
  run_options.mutable_run_options()->set_local_device_count(1);

  std::vector<se::Stream*> async_streams = {async_stream.get()};
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, async_streams,
                               LocalDeviceId(executor->device_ordinal())));

  ASSERT_OK_AND_ASSIGN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(collective_params, collective_config));
  absl::Status is_supported = collective_kernel_thunk->IsSupported(
      clique_key, *executor, collective_params);
  if (!is_supported.ok()) {
    GTEST_SKIP() << "Collective kernel is not supported on this executor: "
                 << is_supported;
  }

  BufferAllocations allocations1({src1, dst1}, 0, nullptr);
  BufferAllocations allocations2({src2, dst2}, 0, nullptr);

  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations1);
  CommandBufferRequests command_buffer_requests;
  Thunk::PrepareParams prepare_params{
      /*collective_params=*/&collective_params,
      /*collective_clique_requests=*/&clique_requests,
      /*collective_memory_requests=*/&memory_requests,
      /*executor=*/executor,
      /*buffer_allocations=*/&allocations1,
      /*execution_scoped_state=*/nullptr,
      /*custom_options=*/nullptr,
      /*command_buffer_requests=*/&command_buffer_requests};
  ASSERT_OK(collective_kernel_thunk->Prepare(prepare_params));

  CollectiveMemoryCache collective_memory_cache(/*num_devices=*/2);
  ASSERT_OK_AND_ASSIGN(
      CollectiveCliques collective_cliques,
      AcquireCollectiveCliques(collective_params, clique_requests));
  ASSERT_OK_AND_ASSIGN(
      CollectiveMemory collective_memory,
      AcquireCollectiveMemory(collective_params, collective_cliques,
                              memory_requests, collective_memory_cache));

  Thunk::InitializeParams initialize_params(&command_buffer_requests);
  initialize_params.executor = executor;
  initialize_params.src.text = kKernelSource;
  initialize_params.buffer_allocations = &allocations1;
  initialize_params.stream = stream.get();
  initialize_params.collective_params = &collective_params;
  initialize_params.collective_cliques = &collective_cliques;
  initialize_params.collective_memory = &collective_memory;
  ASSERT_OK(collective_kernel_thunk->Initialize(initialize_params));
  ASSERT_OK(stream->BlockHostUntilDone());

  Thunk::ExecuteParams params1 = Thunk::ExecuteParams::Create(
      run_options, allocations1, stream.get(), trace_stream.get(),
      &collective_params, &collective_cliques, &collective_memory);

  CommandStateManager state;
  Command::RecordParams record_params = {state};
  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      collective_kernel_thunk->Record(
          params1, record_params, Command::RecordCreate{/*dependencies=*/{}},
          command_buffer.get()));
  ASSERT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());

  BufferAllocations updated_allocations({src2, dst2}, 0, nullptr);
  Thunk::ExecuteParams params2 = Thunk::ExecuteParams::Create(
      run_options, updated_allocations, stream.get(), trace_stream.get(),
      &collective_params, &collective_cliques, &collective_memory);
  std::vector<BufferAllocation::Index> updated_allocs = {0, 1};
  Command::RecordParams update_record_params = {state,
                                                std::move(updated_allocs)};
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* updated_cmd,
                       collective_kernel_thunk->Record(
                           params2, update_record_params,
                           Command::RecordUpdate{cmd}, command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);
  ASSERT_OK(command_buffer->Finalize());
}

class CollectiveKernelThunkCliqueSplitTest
    : public ::testing::TestWithParam<SymmetricMemoryType> {};

TEST_P(CollectiveKernelThunkCliqueSplitTest,
       ReinitializesSymmetricMemoryAfterCliqueSplit) {
  auto [input_data, expected_output_data] =
      MakeSequentialInputAndExpectedOutput();

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/1, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/true,
      /*scratch_memory_type=*/SymmetricMemoryType::kLoadStoreAccessible,
      /*output_memory_type=*/GetParam());

  se::StreamExecutor* executor0 = GetGpuExecutor(0);
  // First execution initializes the 1-device clique [0] as a root clique and
  // ties SymmetricMemory to it.
  ASSERT_OK(RunCollectiveKernelThunk(metadata, executor0, input_data));

  // Now acquire a 2-device parent clique [0, 1] and split [0] and [1] from it.
  // AcquireClique abandons the root clique [0], expiring the thunk's TiedRef
  // and any cached weak_ptr<SymmetricMemory>.
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "split_threads",
                                        /*num_threads=*/2);
    std::vector<tsl::Future<>> futures(2);
    for (int d = 0; d < 2; ++d) {
      futures[d] =
          tsl::MakeFutureOn<>(*thread_pool.AsExecutor(), [d]() -> absl::Status {
            se::StreamExecutor* executor = GetGpuExecutor(d);
            ABSL_ASSIGN_OR_RETURN(auto stream, executor->CreateStream());
            GpuExecutableRunOptions gpu_options;
            gpu_options.set_gpu_global_device_ids(
                GpuExecutableRunOptions::DeviceIdMap{
                    {LocalDeviceId(0), GlobalDeviceId(0)},
                    {LocalDeviceId(1), GlobalDeviceId(1)}});
            DeviceAssignment device_assignment(/*replica_count=*/2,
                                               /*computation_count=*/1);
            device_assignment(0, 0) = 0;
            device_assignment(1, 0) = 1;
            ServiceExecutableRunOptions run_options;
            run_options.mutable_run_options()->set_stream(stream.get());
            run_options.mutable_run_options()->set_device_assignment(
                &device_assignment);
            run_options.mutable_run_options()->set_gpu_executable_run_options(
                &gpu_options);
            ABSL_ASSIGN_OR_RETURN(CollectiveParams collective_params,
                             CollectiveParams::Create(
                                 run_options, /*async_streams=*/{},
                                 LocalDeviceId(executor->device_ordinal())));

            CollectiveCliqueRequests clique_requests;
            GpuCliqueKey parent_key({GlobalDeviceId(0), GlobalDeviceId(1)},
                                    /*num_local_participants=*/2);
            GpuCliqueKey child_key({GlobalDeviceId(d)},
                                   /*num_local_participants=*/1);
            ABSL_RETURN_IF_ERROR(clique_requests.RequestClique(
                parent_key,
                /*device_groups=*/{{GlobalDeviceId(0), GlobalDeviceId(1)}}));
            ABSL_RETURN_IF_ERROR(clique_requests.RequestClique(
                child_key,
                /*device_groups=*/{{GlobalDeviceId(0)}, {GlobalDeviceId(1)}}));
            ABSL_ASSIGN_OR_RETURN(
                CollectiveCliques cliques,
                AcquireCollectiveCliques(collective_params, clique_requests));
            return absl::OkStatus();
          });
    }
    ASSERT_OK(JoinFutures(futures).Await());
  }

  // Second execution on the same thunk must detect the expired references,
  // re-create SymmetricMemory on the newly split clique [0], and update device
  // metadata.
  ASSERT_OK_AND_ASSIGN(
      se::DeviceAddressBase result_buffer2,
      RunCollectiveKernelThunk(metadata, executor0, input_data));
  VerifyDeviceOutputs({result_buffer2}, metadata.input_data_size_bytes,
                      expected_output_data);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveKernelThunkCliqueSplitTest, CollectiveKernelThunkCliqueSplitTest,
    ::testing::Values(SymmetricMemoryType::kNone,
                      SymmetricMemoryType::kLoadStoreAccessible),
    [](const ::testing::TestParamInfo<SymmetricMemoryType>& info) {
      return info.param == SymmetricMemoryType::kNone ? "ScratchOnly"
                                                      : "IoAndScratch";
    });

// ProcessGpuCliques caches cliques in an absl::NoDestructor map keyed by
// (Collectives*, GpuCliqueKey) without destroying them at process exit. Keep
// mock collectives and wrapped communicators alive for the process lifetime,
// mark the mocks leak-allowed for gMock's exit check, and verify all
// expectations explicitly via VerifyAndLeakMockCollectives below.
struct WrappedComm {
  std::unique_ptr<GpuCommunicator> inner;
  StrictMock<MockGpuCommunicator>* mock;
};

struct LeakedMocks {
  std::vector<std::unique_ptr<StrictMock<MockGpuCollectives>>> collectives;
  std::vector<WrappedComm> comms;
};

LeakedMocks& GetLeakedMocks() {
  static absl::NoDestructor<LeakedMocks> leaked_mocks;
  return *leaked_mocks;
}

StrictMock<MockGpuCollectives>* CreateMockCollectives(
    int expected_create_symmetric_memory_calls,
    std::vector<WrappedComm>& wrapped_comms) {
  se::StreamExecutor* executor0 = GetGpuExecutor(0);
  GpuCollectives* default_collectives =
      GpuCollectives::Default(executor0->GetPlatform()->Name());

  auto mock_collectives = std::make_unique<StrictMock<MockGpuCollectives>>();
  StrictMock<MockGpuCollectives>* collectives = mock_collectives.get();
  ::testing::Mock::AllowLeak(collectives);
  GetLeakedMocks().collectives.push_back(std::move(mock_collectives));

  EXPECT_CALL(*collectives, CreateUniqueCliqueId())
      .WillOnce([default_collectives]() {
        return default_collectives->CreateUniqueCliqueId();
      });
  EXPECT_CALL(*collectives, CreateCommunicatorsWithCancel(_, _, _, _, _))
      .WillOnce(
          [default_collectives, expected_create_symmetric_memory_calls,
           &wrapped_comms](const CliqueKey& clique_key,
                           const std::optional<CliqueIds>& clique_ids,
                           absl::Span<const Collectives::DeviceRank> ranks,
                           const Collectives::Config& config,
                           std::shared_ptr<CancellationToken> cancel)
              -> absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> {
            ABSL_ASSIGN_OR_RETURN(
                std::vector<std::unique_ptr<Communicator>> comms,
                default_collectives->CreateCommunicatorsWithCancel(
                    clique_key, clique_ids, ranks, config, cancel));
            std::vector<std::unique_ptr<Communicator>> result;
            result.reserve(comms.size());
            for (std::unique_ptr<Communicator>& comm : comms) {
              std::unique_ptr<GpuCommunicator> inner_comm = absl::WrapUnique(
                  absl::down_cast<GpuCommunicator*>(comm.release()));
              GpuCommunicator* inner_ptr = inner_comm.get();

              auto mock_comm =
                  std::make_unique<StrictMock<MockGpuCommunicator>>();
              ::testing::Mock::AllowLeak(mock_comm.get());
              EXPECT_CALL(*mock_comm, CreateSymmetricMemory(_, _))
                  .Times(expected_create_symmetric_memory_calls)
                  .WillRepeatedly([inner_ptr](se::DeviceAddressBase addr,
                                              bool symmetric_window) {
                    return inner_ptr->CreateSymmetricMemory(addr,
                                                            symmetric_window);
                  });
              EXPECT_CALL(*mock_comm, HealthCheck())
                  .Times(AnyNumber())
                  .WillRepeatedly(
                      [inner_ptr]() { return inner_ptr->HealthCheck(); });
              wrapped_comms.push_back({std::move(inner_comm), mock_comm.get()});
              result.push_back(std::move(mock_comm));
            }
            return result;
          });
  return collectives;
}

void VerifyAndLeakMockCollectives(StrictMock<MockGpuCollectives>* collectives,
                                  std::vector<WrappedComm>& wrapped_comms,
                                  int expected_num_devices) {
  EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(collectives));
  ASSERT_EQ(wrapped_comms.size(), expected_num_devices);
  for (WrappedComm& comm : wrapped_comms) {
    EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(comm.mock));
    GpuCommunicator* inner_ptr = comm.inner.get();
    EXPECT_CALL(*comm.mock, HealthCheck())
        .Times(AnyNumber())
        .WillRepeatedly([inner_ptr]() { return inner_ptr->HealthCheck(); });
    GetLeakedMocks().comms.push_back(std::move(comm));
  }
}

struct SymmetricMemoryCachingTestCase {
  std::string test_name;
  SymmetricMemoryType output_memory_type;
  bool reallocate_on_second_run;
  int expected_create_symmetric_memory_calls;
};

class CollectiveKernelThunkSymmetricMemoryCachingTest
    : public ::testing::TestWithParam<SymmetricMemoryCachingTestCase> {};

TEST_P(CollectiveKernelThunkSymmetricMemoryCachingTest,
       CachesOrUpdatesSymmetricMemoryAcrossRuns) {
  static constexpr int kDevicesCount = 2;
  const SymmetricMemoryCachingTestCase& param = GetParam();

  auto [input_data1, expected_output_data1] =
      MakeSequentialInputAndExpectedOutput(/*base_value=*/0);
  auto [input_data2, expected_output_data2] =
      MakeSequentialInputAndExpectedOutput(/*base_value=*/1000);

  std::vector<WrappedComm> wrapped_comms;
  StrictMock<MockGpuCollectives>* collectives = CreateMockCollectives(
      param.expected_create_symmetric_memory_calls, wrapped_comms);

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/kDevicesCount, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/true,
      /*scratch_memory_type=*/SymmetricMemoryType::kLoadStoreAccessible,
      /*output_memory_type=*/param.output_memory_type);

  ASSERT_OK_AND_ASSIGN(
      std::vector<se::DeviceAddressBase> result_buffers1,
      RunCollectiveKernelThunkOnDevices(
          metadata, /*emulate_multiprocess=*/false, input_data1, collectives));
  VerifyDeviceOutputs(result_buffers1, metadata.input_data_size_bytes,
                      expected_output_data1);

  if (param.reallocate_on_second_run) {
    metadata.device_buffers.assign(kDevicesCount, se::DeviceAddressBase());
  }

  ASSERT_OK_AND_ASSIGN(
      std::vector<se::DeviceAddressBase> result_buffers2,
      RunCollectiveKernelThunkOnDevices(
          metadata, /*emulate_multiprocess=*/false, input_data2, collectives));
  if (param.reallocate_on_second_run) {
    for (int d = 0; d < kDevicesCount; ++d) {
      ASSERT_NE(result_buffers1[d].opaque(), result_buffers2[d].opaque());
    }
    VerifyDeviceOutputs(result_buffers1, metadata.input_data_size_bytes,
                        expected_output_data1);
  }
  VerifyDeviceOutputs(result_buffers2, metadata.input_data_size_bytes,
                      expected_output_data2);

  VerifyAndLeakMockCollectives(collectives, wrapped_comms, kDevicesCount);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveKernelThunkSymmetricMemoryCachingTest,
    CollectiveKernelThunkSymmetricMemoryCachingTest,
    ::testing::Values(
        SymmetricMemoryCachingTestCase{
            "ScratchOnly", SymmetricMemoryType::kNone,
            /*reallocate_on_second_run=*/false,
            /*expected_create_symmetric_memory_calls=*/1},
        SymmetricMemoryCachingTestCase{
            "IoAndScratchSameAddress",
            SymmetricMemoryType::kLoadStoreAccessible,
            /*reallocate_on_second_run=*/false,
            /*expected_create_symmetric_memory_calls=*/2},
        SymmetricMemoryCachingTestCase{
            "IoAndScratchNewAddress", SymmetricMemoryType::kLoadStoreAccessible,
            /*reallocate_on_second_run=*/true,
            /*expected_create_symmetric_memory_calls=*/3}),
    [](const ::testing::TestParamInfo<SymmetricMemoryCachingTestCase>& info) {
      return info.param.test_name;
    });

// Default spec with the given scratch / output symmetric memory types and the
// remote scratch sized for `kNumElements`.
CollectiveKernelSpec MakeDefaultSpec(SymmetricMemoryType scratch_memory_type,
                                     SymmetricMemoryType output_memory_type) {
  return CreateCollectiveKernelSpec(
      kNumElements, /*signal_size=*/1024,
      /*remote_size=*/
      xla::RoundUpTo<uint64_t>(kNumElements * sizeof(uint64_t),
                               kXlaAllocatedBufferAlignBytes),
      /*is_multimem_enabled=*/false, scratch_memory_type, output_memory_type);
}

// Spec where the kernel reads the remote data buffer (scratch 0) directly and
// only the signal buffer (scratch 1) goes through the peer pointer table.
CollectiveKernelSpec MakeNoneScratchSpec() {
  return {
      /*codegen_config=*/{
          /*copy_input_to_scratch=*/false,
          /*input_buffer_specs=*/
          {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
          /*output_buffer_specs=*/
          {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
          /*argument_descriptors=*/
          {{KernelArgType::kInputBuffer, 0},
           {KernelArgType::kOutputBuffer, 0},
           {KernelArgType::kRuntimeRank},
           {KernelArgType::kScratchBuffer, 1},
           {KernelArgType::kScratchBuffer, 0}},
      },
      /*scratch_buffers=*/
      {{/*size=*/kNumElements * sizeof(uint64_t),
        /*requires_multimem=*/false, SymmetricMemoryType::kNone,
        /*should_memzero=*/false, /*should_double_buffer=*/true},
       {/*size=*/1024,
        /*requires_multimem=*/false, SymmetricMemoryType::kLoadStoreAccessible,
        /*should_memzero=*/true, /*should_double_buffer=*/true}},
  };
}

// Illegal spec: mixes a kLoadStoreAccessible output with kXlaRendezvous
// scratch.
CollectiveKernelSpec MakeSymmetricOutputRendezvousScratchSpec() {
  return {
      /*codegen_config=*/{
          /*copy_input_to_scratch=*/false,
          /*input_buffer_specs=*/
          {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
          /*output_buffer_specs=*/
          {{/*requires_multimem=*/false,
            SymmetricMemoryType::kLoadStoreAccessible}},
          /*argument_descriptors=*/
          {{KernelArgType::kInputBuffer, 0},
           {KernelArgType::kOutputBuffer, 0},
           {KernelArgType::kRuntimeRank},
           {KernelArgType::kScratchBuffer, 0},
           {KernelArgType::kScratchBuffer, 1}},
      },
      /*scratch_buffers=*/
      {{/*size=*/1024,
        /*requires_multimem=*/false, SymmetricMemoryType::kXlaRendezvous,
        /*should_memzero=*/true, /*should_double_buffer=*/true},
       {/*size=*/kNumElements * sizeof(uint64_t),
        /*requires_multimem=*/false, SymmetricMemoryType::kNone,
        /*should_memzero=*/false, /*should_double_buffer=*/true}},
  };
}

CollectiveKernelSpec MakeMultimemRendezvousScratchSpec() {
  return CreateCollectiveKernelSpec(
      kNumElements, /*signal_size=*/1024,
      /*remote_size=*/kNumElements * sizeof(uint64_t),
      /*is_multimem_enabled=*/true,
      /*scratch_memory_type=*/SymmetricMemoryType::kXlaRendezvous);
}

struct KernelSpecTestCase {
  std::string test_name;
  std::function<CollectiveKernelSpec()> make_spec;
  // Selects the PTX kernel: kNone -> plain output, otherwise peer-table output.
  SymmetricMemoryType output_memory_type = SymmetricMemoryType::kNone;
  // Put the output slice in a second allocation that aliases allocation 0.
  bool alias_output_allocation = false;
  // If set, run with StrictMock collectives and expect exactly this many
  // CreateSymmetricMemory calls per device.
  std::optional<int> expected_create_symmetric_memory_calls;
  // kOk: outputs are verified. Otherwise the run must fail with this code and
  // message.
  absl::StatusCode expected_code = absl::StatusCode::kOk;
  std::string expected_message_substr;
};

class CollectiveKernelThunkKernelSpecTest
    : public ::testing::TestWithParam<KernelSpecTestCase> {};

TEST_P(CollectiveKernelThunkKernelSpecTest, RunsOrRejectsSpec) {
  static constexpr int kDevicesCount = 2;
  const KernelSpecTestCase& param = GetParam();
  auto [input_data, expected_output_data] =
      MakeSequentialInputAndExpectedOutput();

  std::vector<WrappedComm> wrapped_comms;
  StrictMock<MockGpuCollectives>* collectives = nullptr;
  if (param.expected_create_symmetric_memory_calls.has_value()) {
    collectives = CreateMockCollectives(
        *param.expected_create_symmetric_memory_calls, wrapped_comms);
  }

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/kDevicesCount, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/true,
      /*scratch_memory_type=*/SymmetricMemoryType::kLoadStoreAccessible,
      param.output_memory_type, param.make_spec(),
      param.alias_output_allocation);

  absl::StatusOr<std::vector<se::DeviceAddressBase>> result_buffers =
      RunCollectiveKernelThunkOnDevices(
          metadata, /*emulate_multiprocess=*/false, input_data, collectives);
  if (param.expected_code != absl::StatusCode::kOk) {
    EXPECT_THAT(result_buffers,
                StatusIs(param.expected_code,
                         HasSubstr(param.expected_message_substr)));
    return;
  }
  ASSERT_OK(result_buffers);
  VerifyDeviceOutputs(*result_buffers, metadata.input_data_size_bytes,
                      expected_output_data);
  if (collectives != nullptr) {
    VerifyAndLeakMockCollectives(collectives, wrapped_comms, kDevicesCount);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveKernelThunkKernelSpecTest, CollectiveKernelThunkKernelSpecTest,
    ::testing::Values(
        KernelSpecTestCase{
            "ScratchBufferWithNoneSymmetricMemoryType",
            MakeNoneScratchSpec,
        },
        // Two buffer allocations (index 0 and 1) are backed by the same device
        // buffer and the symmetric output lives in allocation 1. `Prepare`
        // registers symmetric memory by allocation index, so `Initialize` must
        // look it up by slice and not by device address (which resolves to
        // allocation 0).
        KernelSpecTestCase{
            "SymmetricIoBufferResolvedBySliceNotAddress",
            [] {
              return MakeDefaultSpec(SymmetricMemoryType::kLoadStoreAccessible,
                                     SymmetricMemoryType::kLoadStoreAccessible);
            },
            SymmetricMemoryType::kLoadStoreAccessible,
            /*alias_output_allocation=*/true,
        },
        KernelSpecTestCase{
            "SymmetricMultimemOutputBuffer",
            [] {
              return CreateCollectiveKernelSpec(
                  kNumElements, /*signal_size=*/1024,
                  /*remote_size=*/
                  xla::RoundUpTo<uint64_t>(kNumElements * sizeof(uint64_t),
                                           kXlaAllocatedBufferAlignBytes),
                  /*is_multimem_enabled=*/false,
                  SymmetricMemoryType::kLoadStoreAccessible,
                  SymmetricMemoryType::kLoadStoreAccessible,
                  /*output_requires_multimem=*/true);
            },
            SymmetricMemoryType::kLoadStoreAccessible,
        },
        KernelSpecTestCase{
            "PrepareRejectsMixedRendezvousAndLoadStoreAccessible",
            MakeSymmetricOutputRendezvousScratchSpec,
            SymmetricMemoryType::kLoadStoreAccessible,
            /*alias_output_allocation=*/false,
            /*expected_create_symmetric_memory_calls=*/std::nullopt,
            absl::StatusCode::kInternal,
            "must not mix kXlaRendezvous and kLoadStoreAccessible",
        },
        KernelSpecTestCase{
            "PrepareRejectsRendezvousIoBuffer",
            [] {
              return MakeDefaultSpec(SymmetricMemoryType::kXlaRendezvous,
                                     SymmetricMemoryType::kXlaRendezvous);
            },
            SymmetricMemoryType::kXlaRendezvous,
            /*alias_output_allocation=*/false,
            /*expected_create_symmetric_memory_calls=*/std::nullopt,
            absl::StatusCode::kInternal,
            "kXlaRendezvous is not supported for IO buffer specs",
        },
        KernelSpecTestCase{
            "PrepareRejectsMultimemWithoutLoadStoreAccess",
            MakeMultimemRendezvousScratchSpec,
            SymmetricMemoryType::kNone,
            /*alias_output_allocation=*/false,
            /*expected_create_symmetric_memory_calls=*/std::nullopt,
            absl::StatusCode::kInternal,
            "requires_multimem",
        }),
    [](const ::testing::TestParamInfo<KernelSpecTestCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla::gpu
