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
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
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

absl::StatusOr<se::StreamExecutor*> GpuExecutor(int32_t device_ordinal) {
  TF_ASSIGN_OR_RETURN(auto name, PlatformUtil::CanonicalPlatformName("gpu"));
  TF_ASSIGN_OR_RETURN(auto* platform,
                      se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(device_ordinal);
}

TEST(CollectiveKernelThunkTest, ExecutesPtxKernel) {
  using DataT = int64_t;
  static constexpr int64_t kNumElements = 128;
  static constexpr int64_t kInputSizeBytes = kNumElements * sizeof(DataT);
  static constexpr uint32_t kExpectedSignalValue = 1;

  // --------------------
  // Arrange
  // --------------------
  // # Prepare input data and expected output data.
  Array<DataT> input_data({/*num_elements=*/kNumElements});
  input_data.FillRandom(5, 5, /*seed=*/12345);
  Array<DataT> expected_output_data({/*num_elements=*/kNumElements});
  expected_output_data.Each([&](absl::Span<const int64_t> indices, DataT* val) {
    *val = input_data(indices) + kExpectedSignalValue;
  });
  // # Prepare Infrastructure.
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor0, GpuExecutor(0));
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = kProfileName;
  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);
  CollectiveConfig collective_config{
      /*operand_count=*/1,
      /*operand_element_type=*/{PrimitiveType::F32},
      /* replica_groups=*/{replica_group},
      /* collective_op_kind=*/RendezvousKey::CollectiveOpKind::kCrossReplica,
      /* op_id=*/0,
      /* group_mode=*/
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA,
      /* use_symmetric_buffer=*/false};
  const int64_t aligned_input_size_bytes =
      xla::RoundUpTo<uint64_t>(kInputSizeBytes, kXlaAllocatedBufferAlignBytes);
  // 2x because we have two buffers, one for input and one for output so we can
  // test output independently of input.
  const int64_t total_buffer_size = aligned_input_size_bytes * 2;
  // ## Create physical buffers.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor0->CreateStream());
  std::vector<se::DeviceMemoryBase> allocated_buffers = {
      executor0->AllocateArray<DataT>(total_buffer_size)};
  std::vector<se::DeviceMemoryBase> input_buffers = {
      allocated_buffers[0].GetByteSlice(0, aligned_input_size_bytes)};
  std::vector<se::DeviceMemoryBase> output_buffers = {
      allocated_buffers[0].GetByteSlice(aligned_input_size_bytes,
                                        aligned_input_size_bytes)};
  BufferAllocations buffer_allocations(
      /*buffers=*/allocated_buffers,
      /*device_ordinal=*/0,
      /*memory_allocator=*/nullptr);
  TF_ASSERT_OK(
      stream->Memcpy(&input_buffers[0], input_data.data(), kInputSizeBytes));

  // ## Create Logical Buffers.
  BufferAllocation buffer_allocation(
      /*index=*/0, /*size=*/total_buffer_size, /*color=*/0);
  BufferAllocation::Slice input_slice(&buffer_allocation, /*offset=*/0,
                                      /*size=*/aligned_input_size_bytes);
  BufferAllocation::Slice output_slice(
      &buffer_allocation, aligned_input_size_bytes, aligned_input_size_bytes);
  std::vector<CollectiveThunk::Buffer> buffers = {
      {/*element_count=*/kNumElements,
       /*source_buffer=*/input_slice,
       /*destination_buffer=*/output_slice,
       /*source_memory_space=*/0,
       /*destination_memory_space=*/0}};

  // ## Setup device mapping.
  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(
      std::map{std::make_pair(0, GlobalDeviceId(0))});
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);
  TF_ASSERT_OK_AND_ASSIGN(
      auto collective_params,
      Thunk::CollectiveExecuteParams::Create(run_options, /*async_streams=*/{},
                                             /*local_device_ordinal=*/0));
  // --------------------
  // Act
  // --------------------
  CollectiveKernelThunk thunk(std::move(thunk_info), collective_config,
                              ReductionKind::SUM,
                              /*is_async=*/false, buffers,
                              /*is_collective_kernel_enabled=*/true,
                              /*kernel_name=*/kKernelName);

  // # Thunk::Initialize
  Thunk::InitializeParams initialize_params;
  initialize_params.executor = executor0;
  initialize_params.stream = stream.get();
  initialize_params.buffer_allocations = &buffer_allocations;
  initialize_params.collective_params = &collective_params;
  initialize_params.src = {kKernelSource};
  TF_ASSERT_OK(thunk.Initialize(initialize_params));

  // # Thunk::Execute
  auto execute_params =
      Thunk::ExecuteParams::Create(run_options,                              //
                                   buffer_allocations,                       //
                                   stream.get(),                             //
                                   /*command_buffer_trace_stream=*/nullptr,  //
                                   &collective_params,                       //
                                   /*collective_cliques=*/nullptr            //
      );
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));

  // --------------------
  // Assert
  // --------------------
  Array<DataT> output_data({kNumElements});
  TF_ASSERT_OK(
      stream->Memcpy(output_data.data(), output_buffers[0], kInputSizeBytes));
  for (auto i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(expected_output_data(i), output_data(i))
        << "comparison failed at i = " << i;
  }
}

}  // namespace
}  // namespace xla::gpu
