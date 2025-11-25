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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr absl::string_view kProfileName = "test_kernel_profiler";
static constexpr absl::string_view kKernelName = "seven_argument_kernel";

// Test kernel was compiled using following CUDA source:
// __global__ void seven_argument_kernel(void* metadata,                 // 1
//                                       int64_t* input_buffer,          // 2
//                                       int64_t* output_buffer,         // 3
//                                       int64_t num_elements,           // 4
//                                       int64_t num_elements_per_rank,  // 5
//                                       int64_t rank_offset,            // 6
//                                       int64_t signal_value            // 7
// ) {
//   (void)num_elements;
//   (void)num_elements_per_rank;
//   (void)rank_offset;
//   (void)signal_value;
//   (void)metadata;
//   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   for (int i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
//     if (i < num_elements) {
//       output_buffer[i] = input_buffer[i] + signal_value;
//     }
//   }
// }
static constexpr absl::string_view kKernelSource = R"(
  .version 8.7
  .target sm_90
  .address_size 64

  //
  //
  .visible .entry seven_argument_kernel(
  .param .u64 .ptr .align 1 metadata,
  .param .u64 .ptr .align 1 input_buffer,
  .param .u64 .ptr .align 1 output_buffer,
  .param .u64 num_elements,
  .param .u64 num_elements_per_rank,
  .param .u64 rank_offset,
  .param .u64 signal_value
  )
  {
  .reg .pred %p<3>;
  .reg .b32 %r<12>;
  .reg .b64 %rd<16>;

  //
  ld.param.b64 %rd6, [num_elements];
  ld.param.b64 %rd8, [input_buffer];
  cvta.to.global.u64 %rd1, %rd8;
  ld.param.b64 %rd9, [output_buffer];
  cvta.to.global.u64 %rd2, %rd9;
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mad.lo.s32 %r8, %r1, %r2, %r3;
  cvt.s64.s32 %rd15, %r8;
  setp.le.s64 %p1, %rd6, %rd15;
  @%p1 bra $L__BB0_3;
  //
  ld.param.b64 %rd7, [signal_value];
  mov.u32 %r9, %nctaid.x;
  mul.lo.s32 %r4, %r9, %r2;
  add.s32 %r10, %r9, %r1;
  mad.lo.s32 %r11, %r2, %r10, %r3;
  $L__BB0_2: //
  shl.b64 %rd10, %rd15, 3;
  add.s64 %rd11, %rd1, %rd10;
  ld.global.b64 %rd12, [%rd11];
  add.s64 %rd13, %rd12, %rd7;
  add.s64 %rd14, %rd2, %rd10;
  st.global.b64 [%rd14], %rd13;
  cvt.s64.s32 %rd15, %r11;
  setp.gt.s64 %p2, %rd6, %rd15;
  add.s32 %r11, %r11, %r4;
  @%p2 bra $L__BB0_2;
  $L__BB0_3:
  ret;
  //
  }
)";

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

struct CollectiveKernelThunkMetadata {
  BufferAllocation buffer_allocation;
  std::unique_ptr<CollectiveKernelThunk> thunk;
  int64_t total_buffer_size;
  int64_t input_data_size_bytes;
  int64_t aligned_input_size_bytes;
  int64_t num_devices;
  std::vector<CollectiveThunk::Buffer> buffers;
};

CollectiveKernelThunkMetadata CreateCollectiveKernelThunk(
    int num_devices, int num_elements, bool is_multimem_enabled) {
  const int64_t input_size_bytes = num_elements * sizeof(uint64_t);
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
  BufferAllocation::Slice input_slice(&result.buffer_allocation, /*offset=*/0,
                                      /*size=*/aligned_input_size_bytes);
  BufferAllocation::Slice output_slice(&result.buffer_allocation,
                                       aligned_input_size_bytes,
                                       aligned_input_size_bytes);
  result.buffers = {{/*element_count=*/num_elements,
                     /*source_buffer=*/input_slice,
                     /*destination_buffer=*/output_slice,
                     /*source_memory_space=*/0,
                     /*destination_memory_space=*/0}};
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = kProfileName;
  result.thunk = std::make_unique<CollectiveKernelThunk>(
      std::move(thunk_info), collective_config, ReductionKind::SUM,
      /*is_async=*/false, result.buffers,
      /*is_collective_kernel_enabled=*/true,
      /*kernel_name=*/kKernelName,
      /*is_multimem_enabled=*/is_multimem_enabled);
  result.total_buffer_size = total_buffer_size;
  result.num_devices = num_devices;
  result.aligned_input_size_bytes = aligned_input_size_bytes;
  result.input_data_size_bytes = input_size_bytes;
  return result;
}

absl::StatusOr<se::DeviceMemoryBase> RunCollectiveKernelThunk(
    CollectiveKernelThunkMetadata& metadata, se::StreamExecutor* executor,
    std::vector<uint64_t> input_data) {
  BufferAllocation buffer_allocation(
      /*index=*/0, /*size=*/metadata.total_buffer_size, /*color=*/0);
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(
      std::map{std::make_pair(0, GlobalDeviceId(0)),
               std::make_pair(1, GlobalDeviceId(1))});

  TF_ASSIGN_OR_RETURN(auto stream, executor->CreateStream());
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

  TF_ASSIGN_OR_RETURN(auto collective_params,
                      CollectiveParams::Create(
                          run_options, /*async_streams=*/{},
                          /*local_device_ordinal=*/executor->device_ordinal()));
  std::vector<se::DeviceMemoryBase> allocated_buffers = {
      executor->AllocateArray<uint64_t>(metadata.total_buffer_size)};

  se::DeviceMemoryBase input_buffer =
      allocated_buffers[0].GetByteSlice(0, metadata.aligned_input_size_bytes);
  se::DeviceMemoryBase output_buffer = allocated_buffers[0].GetByteSlice(
      metadata.aligned_input_size_bytes, metadata.aligned_input_size_bytes);
  BufferAllocations buffer_allocations(
      /*buffers=*/allocated_buffers,
      /*device_ordinal=*/executor->device_ordinal(),
      /*memory_allocator=*/nullptr);

  if (!input_data.empty()) {
    VLOG(3) << "Copying input data to the device";
    TF_RETURN_IF_ERROR(stream->Memcpy(&input_buffer, input_data.data(),
                                      metadata.input_data_size_bytes));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  }

  Thunk::InitializeParams initialize_params;
  initialize_params.executor = executor;
  initialize_params.stream = stream.get();
  initialize_params.buffer_allocations = &buffer_allocations;
  initialize_params.collective_params = &collective_params;
  initialize_params.src = {kKernelSource};
  TF_RETURN_IF_ERROR(metadata.thunk->Initialize(initialize_params));

  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr);
  TF_RETURN_IF_ERROR(metadata.thunk->ExecuteOnStream(execute_params));
  return output_buffer;
}

std::vector<absl::StatusOr<se::DeviceMemoryBase>>
RunCollectiveKernelThunkOnDevices(CollectiveKernelThunkMetadata& metadata) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "device_threads",
                                      metadata.num_devices);
  std::vector<tsl::Future<se::DeviceMemoryBase>> futures;
  for (int device_number = 0; device_number < metadata.num_devices;
       ++device_number) {
    futures.push_back(tsl::Future<se::DeviceMemoryBase>::MakeOn(
        *thread_pool.AsExecutor(), [&metadata, device_number] {
          return RunCollectiveKernelThunk(metadata,
                                          GetGpuExecutor(device_number), {});
        }));
  }

  std::vector<absl::StatusOr<se::DeviceMemoryBase>> results;
  for (auto& future : futures) {
    results.push_back(std::move(future).Await());
  }
  return results;
}

TEST(CollectiveKernelThunkTest, ExecutesPtxKernel) {
  static constexpr int64_t kNumElements = 128;
  static constexpr uint32_t kExpectedSignalValue = 1;

  std::vector<uint64_t> input_data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    input_data[i] = i;
  }

  std::vector<uint64_t> expected_output_data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    expected_output_data[i] = input_data[i] + kExpectedSignalValue;
  }

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/1, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false);

  se::StreamExecutor* executor0 = GetGpuExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemoryBase result_buffer,
      RunCollectiveKernelThunk(metadata, executor0, input_data));

  std::vector<uint64_t> output_data(kNumElements);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor0->CreateStream());
  TF_ASSERT_OK(stream->Memcpy(output_data.data(), result_buffer,
                              metadata.input_data_size_bytes));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  for (auto i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(expected_output_data[i], output_data[i])
        << "comparison failed at i = " << i;
  }
}

TEST(CollectiveKernelThunkTest, MultimemSetupTest) {
  static constexpr int kDevicesCount = 2;
  static constexpr int64_t kNumElements = 128;

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/kDevicesCount, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/true);
  for (absl::StatusOr<se::DeviceMemoryBase> result :
       RunCollectiveKernelThunkOnDevices(metadata)) {
    TF_ASSERT_OK(result);
  }
}

}  // namespace
}  // namespace xla::gpu
