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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/ptx_compile_options_from_debug_options.h"
#include "xla/service/service_executable_run_options.h"
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
#include "xla/stream_executor/memory_space.h"
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
using ::absl_testing::StatusIs;

static constexpr absl::string_view kProfileName = "test_kernel_profiler";
static constexpr absl::string_view kKernelName = "six_argument_kernel";
static constexpr int64_t kNumElements = 128;

// Test kernel was compiled using following CUDA source:
// __global__ void six_argument_kernel(int64_t* input_buffer,          // 1
//                                     int64_t* output_buffer,         // 2
//                                     int64_t rank,                   // 3
//                                     int64_t signal_value            // 4
//                                     int64_t* signal_buffers,        // 5
//                                     int64_t* remote_buffers,        // 6
// ) {
//   (void)rank;
//   (void)signal_buffers;
//   (void)remote_buffers;
//   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   for (int i = idx; i < kNumElements; i += gridDim.x * blockDim.x) {
//     if (i < kNumElements) {
//       output_buffer[i] = input_buffer[i] + signal_value;
//     }
//   }
// }
static constexpr absl::string_view kKernelSource = R"(
  .version 8.7
  .target sm_90
  .address_size 64

  .visible .entry six_argument_kernel(
  .param .u64 .ptr .align 1 input_buffer,
  .param .u64 .ptr .align 1 output_buffer,
  .param .u64 rank,
  .param .u64 signal_value,
  .param .u64 .ptr .align 1 signal_buffers,
  .param .u64 .ptr .align 1 remote_buffers
  )
  {
  .reg .pred %p<3>;
  .reg .b32 %r<7>;
  .reg .b64 %rd<11>;

  ld.param.b64 %rd4, [input_buffer];
  cvta.to.global.u64 %rd1, %rd4;
  ld.param.b64 %rd5, [output_buffer];
  cvta.to.global.u64 %rd2, %rd5;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r1, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r6, %r3, %r1, %r4;
  setp.gt.s32 %p1, %r6, 127;
  @%p1 bra $L__BB0_3;
  //
  ld.param.b64 %rd3, [signal_value];
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
  // If true, the PTX is not compiled into CUBIN and is passed to the thunk as
  // a string.
  bool use_ptx;
  std::vector<CollectiveThunk::Buffer> buffers;
};

CollectiveKernelThunkMetadata CreateCollectiveKernelThunk(
    int num_devices, int num_elements, bool is_multimem_enabled, bool use_ptx) {
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
  BufferAllocation::Slice input_slice(&result.buffer_allocation, /*offset=*/0,
                                      /*size=*/aligned_input_size_bytes);
  BufferAllocation::Slice output_slice(&result.buffer_allocation,
                                       aligned_input_size_bytes,
                                       aligned_input_size_bytes);
  result.buffers = {{/*element_count=*/num_elements,
                     /*source_buffer=*/{input_slice, input_shape},
                     /*destination_buffer=*/{output_slice, input_shape},
                     /*source_memory_space=*/0,
                     /*destination_memory_space=*/0}};
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = kProfileName;
  const LaunchDimensions launch_dimensions(
      /*block_x_count=*/1, /*thread_x_count_per_block=*/kNumElements);
  result.thunk = std::make_unique<CollectiveKernelThunk>(
      std::move(thunk_info), collective_config, ReductionKind::SUM,
      /*is_async=*/false, result.buffers,
      /*is_collective_kernel_enabled=*/true,
      /*kernel_name=*/kKernelName,
      /*launch_dimensions=*/launch_dimensions,
      /*shmem_bytes=*/0,
      /*is_multimem_enabled=*/is_multimem_enabled);
  result.total_buffer_size = total_buffer_size;
  result.num_devices = num_devices;
  result.aligned_input_size_bytes = aligned_input_size_bytes;
  result.input_data_size_bytes = input_size_bytes;
  result.use_ptx = use_ptx;
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
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::cuda::CompilationProvider> compilation_provider,
      se::cuda::AssembleCompilationProvider(options));
  se::CudaComputeCapability cc =
      *device_description.gpu_compute_capability().cuda_compute_capability();
  se::cuda::CompilationOptions compilation_options =
      PtxCompileOptionsFromDebugOptions(debug_options);
  TF_ASSIGN_OR_RETURN(
      se::cuda::Assembly assembly,
      compilation_provider->Compile(cc, ptx_string, compilation_options));
  return std::move(assembly.cubin);
}

absl::StatusOr<se::DeviceAddressBase> RunCollectiveKernelThunk(
    CollectiveKernelThunkMetadata& metadata, se::StreamExecutor* executor,
    std::vector<uint64_t> input_data, bool emulate_multiprocess = false) {
  BufferAllocation buffer_allocation(
      /*index=*/0, /*size=*/metadata.total_buffer_size, /*color=*/0);
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0)),
      std::make_pair(LocalDeviceId(1), GlobalDeviceId(1))});

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

  TF_ASSIGN_OR_RETURN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  // We always allocate from collective memory space because we want to be able
  // to test multimem kernels. In XLA programs it's up to the compiler to assign
  // correct memory space to kernel parameters and results buffers.
  std::vector<se::DeviceAddressBase> allocated_buffers = {
      executor->AllocateArray<uint64_t>(
          metadata.total_buffer_size,
          /*memory_space=*/static_cast<int>(se::MemorySpace::kCollective))};

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
    TF_RETURN_IF_ERROR(stream->Memcpy(&input_buffer, input_data.data(),
                                      metadata.input_data_size_bytes));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  }

  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(buffer_allocations);
  Thunk::PrepareParams prepare_params{&collective_params, &clique_requests,
                                      &memory_requests,   nullptr,
                                      executor,           &buffer_allocations};

  TF_RETURN_IF_ERROR(metadata.thunk->Prepare(prepare_params));
  TF_ASSIGN_OR_RETURN(CollectiveMemory collective_memory,
                      AcquireCollectiveMemory(collective_params, /*cliques=*/{},
                                              memory_requests));

  Thunk::InitializeParams initialize_params;
  initialize_params.executor = executor;
  initialize_params.stream = stream.get();
  initialize_params.buffer_allocations = &buffer_allocations;
  initialize_params.collective_params = &collective_params;
  initialize_params.src = {kKernelSource};
  initialize_params.collective_memory = &collective_memory;

  GpuExecutableRunOptions::DeviceIdMap global_device_id_map = {
      {LocalDeviceId(0), GlobalDeviceId(0)}};
  if (emulate_multiprocess) {
    initialize_params.collective_params->global_device_id_map =
        &global_device_id_map;
  }

  std::vector<uint8_t> cubin;
  if (!metadata.use_ptx) {
    TF_ASSIGN_OR_RETURN(
        cubin,
        CompilePtxToCubin(kKernelSource, executor->GetDeviceDescription(),
                          DebugOptions()));
    initialize_params.src.binary = cubin;
  }
  TF_RETURN_IF_ERROR(metadata.thunk->Initialize(initialize_params));

  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr, &collective_params,
      /*collective_cliques=*/nullptr, /*collective_memory=*/&collective_memory);
  TF_RETURN_IF_ERROR(metadata.thunk->ExecuteOnStream(execute_params));
  return output_buffer;
}

absl::StatusOr<std::vector<se::DeviceAddressBase>>
RunCollectiveKernelThunkOnDevices(CollectiveKernelThunkMetadata& metadata,
                                  bool emulate_multiprocess = false) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "device_threads",
                                      metadata.num_devices);

  std::vector<tsl::Future<se::DeviceAddressBase>> futures(metadata.num_devices);
  for (int d = 0; d < metadata.num_devices; ++d) {
    futures[d] = tsl::MakeFutureOn<se::DeviceAddressBase>(
        *thread_pool.AsExecutor(), [&metadata, d, emulate_multiprocess] {
          return RunCollectiveKernelThunk(metadata, GetGpuExecutor(d), {},
                                          emulate_multiprocess);
        });
  }

  return JoinFutures<se::DeviceAddressBase>(futures).Await();
}

class CollectiveKernelThunkParameterizedTest
    : public ::testing::TestWithParam<bool> {};

TEST_P(CollectiveKernelThunkParameterizedTest, ExecutesPtxKernel) {
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
      /*is_multimem_enabled=*/false, /*use_ptx=*/GetParam());

  se::StreamExecutor* executor0 = GetGpuExecutor(0);
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceAddressBase result_buffer,
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
  TF_ASSERT_OK(RunCollectiveKernelThunkOnDevices(metadata));
}

TEST(CollectiveKernelThunkTest, MultiprocessTest) {
  static constexpr int kDevicesCount = 2;

  CollectiveKernelThunkMetadata metadata = CreateCollectiveKernelThunk(
      /*num_devices=*/kDevicesCount, /*num_elements=*/kNumElements,
      /*is_multimem_enabled=*/false, /*use_ptx=*/true);
  EXPECT_THAT(RunCollectiveKernelThunkOnDevices(metadata,
                                                /*emulate_multiprocess=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::gpu
