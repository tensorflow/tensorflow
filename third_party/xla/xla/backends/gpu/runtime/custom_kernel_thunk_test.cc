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

#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {
using absl_testing::IsOkAndHolds;
using ::testing::Field;
using ::testing::Optional;
using tsl::proto_testing::EqualsProto;
using tsl::proto_testing::ParseTextProtoOrDie;

TEST(CustomKernelThunkTest, ToStringIncludesPdlOnlyWhenEnabled) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "test_kernel_profiler";

  emitters::KernelArguments kernel_arguments({});
  GpuKernelSpec kernel_spec{
      "name",
      LaunchDimensions{se::BlockDim(3, 2, 1), se::ThreadDim(4, 5, 6)},
      kernel_arguments,
      /*scratch_bytes=*/42,
  };
  CustomKernel kernel(kernel_spec,
                      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                          /*ptx=*/"", /*kernel_name=*/"name", /*arity=*/0));
  CustomKernelThunk thunk(thunk_info, kernel, kernel_arguments,
                          /*devices_in_process=*/1);
  EXPECT_EQ(thunk.ToString(0),
            "kernel=name, profile_annotation=test_kernel_profiler, "
            "launch dimensions=blocks: {3, 2, 1}, threads/block: {4, 5, 6}, "
            "cluster_dim=nullopt, shmem_bytes=42");

  GpuKernelSpec clustered_kernel_spec{
      "clustered_name",
      LaunchDimensions{se::BlockDim(6, 4, 2), se::ThreadDim(8, 10, 12)},
      emitters::KernelArguments{std::vector<emitters::KernelArgument>{}},
      /*scratch_bytes=*/84,
      false,
      {},
      se::ClusterDim(2, 2, 2)};
  CustomKernel clustered_kernel(
      clustered_kernel_spec,
      se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
          /*ptx=*/"", /*kernel_name=*/"clustered_name", /*arity=*/0));
  CustomKernelThunk pdl_thunk(thunk_info, clustered_kernel, kernel_arguments,
                              /*devices_in_process=*/1, /*use_pdl=*/true);
  EXPECT_EQ(pdl_thunk.ToString(0),
            "kernel=clustered_name, profile_annotation=test_kernel_profiler, "
            "launch dimensions=blocks: {6, 4, 2}, threads/block: {8, 10, 12}, "
            "cluster_dim=ClusterDim{2, 2, 2}, shmem_bytes=84, use_pdl=true");
}

TEST(CustomKernelThunkTest, BufferUsesReturnsCorrectBuffers) {
  Shape arg_shape = ShapeUtil::MakeShape(F32, {512});
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/512, /*size=*/512);
  emitters::KernelArgument arg0(arg_shape, slice0);
  emitters::KernelArgument arg1(arg_shape, slice1);
  arg0.set_written(false);
  arg1.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0, arg1});

  GpuKernelSpec spec{
      "",
      LaunchDimensions{},
      kernel_arguments,
  };
  CustomKernel kernel(spec, se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                                /*ptx=*/"", /*kernel_name=*/"", /*arity=*/0));
  CustomKernelThunk thunk(Thunk::ThunkInfo{}, kernel, kernel_arguments,
                          /*devices_in_process=*/1);

  Thunk::BufferUses buffers = thunk.buffer_uses();

  ASSERT_THAT(buffers, testing::UnorderedElementsAre(
                           BufferUse::Read(slice0, arg_shape),
                           BufferUse::Write(slice1, arg_shape)));
}

TEST(CustomKernelThunkTest, BufferUsesReturnsBuffersInConsistentOrder) {
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/512, /*size=*/512);
  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {512}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {512}), slice1);
  arg0.set_written(false);
  arg1.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0, arg1});

  GpuKernelSpec spec{
      /*name=*/"",
      LaunchDimensions{},
      kernel_arguments,
  };
  CustomKernel kernel(spec, se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                                /*ptx=*/"", /*kernel_name=*/"", /*arity=*/0));
  CustomKernelThunk thunk(Thunk::ThunkInfo{}, kernel, kernel_arguments,
                          /*devices_in_process=*/1);

  Thunk::BufferUses buffers1 = thunk.buffer_uses();
  Thunk::BufferUses buffers2 = thunk.buffer_uses();

  ASSERT_THAT(buffers1, testing::ContainerEq(buffers2));
}

TEST(CustomKernelThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.thunk_id = 42;

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/512);
  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {512}), slice0);
  arg0.set_written(true);
  emitters::KernelArguments kernel_arguments({arg0});

  GpuKernelSpec spec{
      "name",
      LaunchDimensions{se::BlockDim(3, 2, 1), se::ThreadDim(4, 5, 6)},
      kernel_arguments,
      42,
  };
  CustomKernel kernel(spec, se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                                "PTX", "kernel_name", /*arity=*/1));
  CustomKernelThunk thunk(thunk_info, kernel, kernel_arguments,
                          /*devices_in_process=*/1);

  EXPECT_THAT(
      thunk.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
        thunk_info { profile_annotation: "profile_annotation" thunk_id: 42 }
        custom_kernel_thunk {
          custom_kernel {
            name: "name"
            kernel_spec {
              kernel_name: "kernel_name"
              ptx { data: "PTX" }
              arity: 1
            }
            block_dims { coordinates { x: 3, y: 2, z: 1 } }
            thread_dims { coordinates { x: 4, y: 5, z: 6 } }
            shared_memory_bytes: 42
          }
          args {
            slice { buffer_allocation_index: 0, offset: 0, size: 512 }
            shape {
              element_type: F32
              dimensions: 512
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
              is_dynamic_dimension: false
            }
          }
          written: true
          tma_metadata {}
        }
      )pb")));
}

TEST(CustomKernelThunkTest, FromProto) {
  CustomKernelThunkProto proto = ParseTextProtoOrDie<CustomKernelThunkProto>(
      R"pb(
        custom_kernel {
          name: "test_kernel"
          kernel_spec {
            ptx { data: "PTX" }
            arity: 1
          }
          block_dims { coordinates { x: 1, y: 1, z: 1 } }
          thread_dims { coordinates { x: 1, y: 1, z: 1 } }
          shared_memory_bytes: 42
        }
        args {
          slice { buffer_allocation_index: 0, offset: 0, size: 1024 }
          shape {
            element_type: U8,
            dimensions: 1024,
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
        written: true
      )pb");

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/1024, /*color=*/0);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CustomKernelThunk> thunk,
                       CustomKernelThunk::FromProto(Thunk::ThunkInfo{}, proto,
                                                    buffer_allocations,
                                                    /*devices_in_process=*/1));

  EXPECT_THAT(thunk->custom_kernel().name(), "test_kernel");
  EXPECT_THAT(thunk->arguments(),
              testing::ElementsAre(ShapedSlice{
                  BufferAllocation::Slice(&buffer_allocations[0], /*offset=*/0,
                                          /*size=*/1024),
                  ShapeUtil::MakeShape(U8, {1024})}));
  EXPECT_THAT(thunk->written(), testing::ElementsAre(true));
  EXPECT_THAT(thunk->custom_kernel().kernel_spec().cuda_ptx_in_memory(),
              Optional(Field(&se::CudaPtxInMemory::ptx, "PTX")));
}

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  ABSL_ASSIGN_OR_RETURN(std::string name,
                   PlatformUtil::CanonicalPlatformName("gpu"));
  ABSL_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(absl::AsciiStrToUpper(name)));
  return platform->ExecutorForDevice(0);
}

// Wraps the AddI32 PTX kernel (c[0] = a[0] + b[0]) as a CustomKernel and
// returns a CustomKernelThunk bound to the given three allocations. The kernel
// spec includes a custom args-packing function (required by command buffer
// CreateLaunch/UpdateLaunch with KernelArgsDeviceAddressArray).
static absl::StatusOr<std::unique_ptr<CustomKernelThunk>>
MakeAddI32CustomKernelThunk(
    const std::vector<BufferAllocation>& allocs,
    std::vector<int64_t> zeroed_output_buffer_indices = {}) {
  absl::string_view ptx =
      se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;

  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);

  GpuKernelSpec spec{
      /*kernel_name=*/"AddI32",
      LaunchDimensions{se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)},
      emitters::KernelArguments({arg_a, arg_b, arg_c})};
  ABSL_ASSIGN_OR_RETURN(CustomKernel kernel, kernel::GetPtxCustomKernel(spec, ptx));

  return std::make_unique<CustomKernelThunk>(
      Thunk::ThunkInfo(), std::move(kernel),
      emitters::KernelArguments({arg_a, arg_b, arg_c}),
      /*devices_in_process=*/1, /*use_pdl=*/false,
      std::move(zeroed_output_buffer_indices));
}

TEST(CustomKernelThunkTest, RecordCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 1, val_b = 2;
  ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // 1 + 2 = 3
}

TEST(CustomKernelThunkTest, RecordCommandBufferUpdate) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 10, val_b = 20;
  ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // First recording: RecordCreate.
  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 30);  // 10 + 20 = 30

  // Update and re-submit with same allocations.
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(execute_params, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 30);  // still 10 + 20 = 30
}

TEST(CustomKernelThunkTest, RecordCommandBufferUpdateWithNewOutputBuffer) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c2_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 1, val_b = 2;
  ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c2_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs));

  se::StreamExecutorAddressAllocator allocator(executor);

  BufferAllocations alloc1({a_dev, b_dev, c_dev}, 0, &allocator);
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &alloc1;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params1 =
      Thunk::ExecuteParams::Create(run_options, alloc1, stream.get(), nullptr,
                                   nullptr, nullptr, nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params1, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // 1 + 2 = 3

  // Update to use c2_dev as the output buffer.
  BufferAllocations alloc2({a_dev, b_dev, c2_dev}, 0, &allocator);
  auto execute_params2 =
      Thunk::ExecuteParams::Create(run_options, alloc2, stream.get(), nullptr,
                                   nullptr, nullptr, nullptr, {});

  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(execute_params2, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node reused
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c2_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // 1 + 2 = 3 written into the new buffer

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  EXPECT_EQ(result, 3);  // original buffer still holds first-run result
}

// Zeroed-output coverage for CustomKernelThunk.
//
// AddI32 computes c = a + b and overwrites c, so marking c as zeroed would
// prove nothing. Mark input b instead: nothing requires a zeroed index to be an
// output (the thunk just zeroes args_[index].slice), so this exercises the
// mechanism with an unambiguous signal -- zeroing honoured gives c == 1,
// zeroing dropped gives c == 3.
TEST(CustomKernelThunkTest, ExecuteOnStreamZeroed) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 1, val_b = 2;
  ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs, {1}));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  ASSERT_OK(thunk->ExecuteOnStream(execute_params));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result, 1);  // b zeroed before the launch.
}

TEST(CustomKernelThunkTest, RecordCommandBufferZeroed) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 1, val_b = 2;
  ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs, {1}));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result, 1);
}

TEST(CustomKernelThunkTest, RecordCommandBufferUpdateZeroed) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 1, val_b = 2;
  ASSERT_OK(stream->Memcpy(&a_dev, &val_a, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(stream->MemZero(&c_dev, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs, {1}));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &buffer_allocations;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk->Record(execute_params, record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result, 1) << "RecordCreate";

  // Dirty b again and update; the memset must be re-recorded, not dropped.
  ASSERT_OK(stream->Memcpy(&b_dev, &val_b, sizeof(int32_t)));
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk->Record(execute_params, record_params, Command::RecordUpdate{cmd},
                    command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);
  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result = 0;
  ASSERT_OK(stream->Memcpy(&result, c_dev, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result, 1) << "RecordUpdate";
}

// One thunk recorded twice into the same command buffer, then both recordings
// updated against moved buffers: the shape an unrolled while loop produces.
// Both recordings share one command-state entry, so each update has to reach
// the memset recorded alongside its own node. If it reaches the other one, the
// first recording reads an unzeroed b and c == 3.
TEST(CustomKernelThunkTest, RecordCommandBufferTwoRecordingsZeroed) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // One operand set per recording, plus a fresh b buffer for each so that the
  // update genuinely moves the memset destinations.
  se::DeviceAddress<int32_t> a0 = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b0 = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c0 = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> a1 = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b1 = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c1 = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b0_new = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b1_new = executor->AllocateArray<int32_t>(1, 0);

  int32_t val_a = 1, val_b = 2;
  for (se::DeviceAddress<int32_t>* a : {&a0, &a1}) {
    ASSERT_OK(stream->Memcpy(a, &val_a, sizeof(int32_t)));
  }
  for (se::DeviceAddress<int32_t>* b : {&b0, &b1, &b0_new, &b1_new}) {
    ASSERT_OK(stream->Memcpy(b, &val_b, sizeof(int32_t)));
  }
  for (se::DeviceAddress<int32_t>* c : {&c0, &c1}) {
    ASSERT_OK(stream->MemZero(c, sizeof(int32_t)));
  }

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs, {1}));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations_0({a0, b0, c0}, 0, &allocator);
  BufferAllocations allocations_1({a1, b1, c1}, 0, &allocator);
  BufferAllocations allocations_0_new({a0, b0_new, c0}, 0, &allocator);
  BufferAllocations allocations_1_new({a1, b1_new, c1}, 0, &allocator);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.buffer_allocations = &allocations_0;
  init_params.stream = stream.get();
  ASSERT_OK(thunk->Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto params_for = [&](const BufferAllocations& allocations) {
    return Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                        nullptr, nullptr, nullptr, nullptr, {});
  };

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd_0,
                       thunk->Record(params_for(allocations_0), record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd_1,
                       thunk->Record(params_for(allocations_1), record_params,
                                     Command::RecordCreate{/*dependencies=*/{}},
                                     command_buffer.get()));
  ASSERT_NE(cmd_0, nullptr);
  ASSERT_NE(cmd_1, nullptr);
  ASSERT_NE(cmd_0, cmd_1);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));

  int32_t result_0 = 0, result_1 = 0;
  ASSERT_OK(stream->Memcpy(&result_0, c0, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&result_1, c1, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result_0, 1) << "first recording, RecordCreate";
  EXPECT_EQ(result_1, 1) << "second recording, RecordCreate";

  // Point both recordings at their new b buffers. Each update must reach the
  // memset recorded alongside its own node.
  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_0,
      thunk->Record(params_for(allocations_0_new), record_params,
                    Command::RecordUpdate{cmd_0}, command_buffer.get()));
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_1,
      thunk->Record(params_for(allocations_1_new), record_params,
                    Command::RecordUpdate{cmd_1}, command_buffer.get()));
  EXPECT_EQ(updated_0, cmd_0);
  EXPECT_EQ(updated_1, cmd_1);
  ASSERT_OK(command_buffer->Finalize());

  for (se::DeviceAddress<int32_t>* c : {&c0, &c1}) {
    ASSERT_OK(stream->MemZero(c, sizeof(int32_t)));
  }
  ASSERT_OK(command_buffer->Submit(stream.get()));

  result_0 = 0;
  result_1 = 0;
  ASSERT_OK(stream->Memcpy(&result_0, c0, sizeof(int32_t)));
  ASSERT_OK(stream->Memcpy(&result_1, c1, sizeof(int32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(result_0, 1) << "first recording, RecordUpdate";
  EXPECT_EQ(result_1, 1) << "second recording, RecordUpdate";
}

TEST(CustomKernelThunkTest, RecordFailsWithoutInitialize) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  ASSERT_OK_AND_ASSIGN(auto thunk, MakeAddI32CustomKernelThunk(allocs));
  // Intentionally skip Initialize().

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  auto status = thunk->Record(execute_params, record_params,
                              Command::RecordCreate{/*dependencies=*/{}},
                              command_buffer.get());
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.status().message(),
              ::testing::HasSubstr("Custom kernel not loaded"));
}

TEST(CustomKernelThunkTest, ExecuteFailsWithoutInitialize) {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  auto name = absl::AsciiStrToUpper(platform_name);
  if (name == "ROCM" || name == "SYCL") {
    GTEST_SKIP() << "AddI32 PTX kernel not supported on ROCm or oneAPI.";
  }
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceAddress<int32_t> a_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> b_dev = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceAddress<int32_t> c_dev = executor->AllocateArray<int32_t>(1, 0);

  int32_t sentinel = 99;
  ASSERT_OK(stream->Memcpy(&c_dev, &sentinel, sizeof(int32_t)));

  std::vector<BufferAllocation> allocs = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),
  };
  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[0], 0, 4));
  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[1], 0, 4));
  emitters::KernelArgument arg_c(ShapeUtil::MakeShape(S32, {1}),
                                 BufferAllocation::Slice(&allocs[2], 0, 4));
  arg_a.set_written(false);
  arg_b.set_written(false);
  arg_c.set_written(true);
  GpuKernelSpec spec{
      /*kernel_name=*/"AddI32",
      LaunchDimensions{se::BlockDim(1, 1, 1), se::ThreadDim(1, 1, 1)},
      emitters::KernelArguments({arg_a, arg_b, arg_c})};

  absl::string_view ptx =
      se::gpu::GetAddI32PtxKernelSpec().cuda_ptx_in_memory().value().ptx;
  ASSERT_OK_AND_ASSIGN(CustomKernel kernel,
                       kernel::GetPtxCustomKernel(spec, ptx));

  CustomKernelThunk thunk(Thunk::ThunkInfo(), std::move(kernel),
                          emitters::KernelArguments({arg_a, arg_b, arg_c}),
                          /*devices_in_process=*/1,
                          /*use_pdl=*/false,
                          /*zeroed_output_buffer_indices=*/{2});
  // Intentionally skip Initialize().

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({a_dev, b_dev, c_dev}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  auto execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(), nullptr, nullptr, nullptr,
      nullptr, {});

  auto status = thunk.ExecuteOnStream(execute_params);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Custom kernel not loaded"));

  int32_t observed = 0;
  ASSERT_OK(stream->Memcpy(&observed, c_dev, sizeof(int32_t)));
  EXPECT_EQ(observed, sentinel);
}

}  // namespace
}  // namespace xla::gpu
