/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/address_computation_thunk.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/gpu_types.h"  // IWYU pragma: keep
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#if GOOGLE_CUDA
#define PLATFORM "CUDA"
#elif TENSORFLOW_USE_ROCM
#define PLATFORM "ROCM"
#endif

namespace xla::gpu {

namespace {

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

}  // namespace

TEST(AddressComputationThunkTest, SlicedGemm) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/0, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs(fake_allocations.back().get(), 0,
                                    rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out(fake_allocations.back().get(), 0,
                                    out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace(fake_allocations.back().get(), 0,
                                          1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/4, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/5, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs, slice_out,
      slice_workspace, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
       std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
       std::nullopt, std::nullopt},
      {sizeof(int64_t), std::nullopt, std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {lhs, rhs, out, workspace, lhs_offset_0, lhs_offset_1}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

TEST(AddressComputationThunkTest, SlicedNonContiguousGemm) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 2;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(float) * 2 * 2;

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/0, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/1, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);

  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out(fake_allocations.back().get(), 0,
                                    out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace(fake_allocations.back().get(), 0,
                                          1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/4, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/5, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_rhs_offset_0(/*index=*/6, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_rhs_offset_0(&alloc_rhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_rhs_offset_1(/*index=*/7, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_rhs_offset_1(&alloc_rhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs_fake,
      slice_out, slice_workspace, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  std::vector<AddressComputationThunk::Offset> rhs_offsets{slice_rhs_offset_0,
                                                           slice_rhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, rhs_offsets, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}),
       ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), std::nullopt,
       std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), std::nullopt,
       std::nullopt},
      {sizeof(int64_t), sizeof(int64_t), std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[2,2]{1,0} slice(lhs), slice={[0:2], [1:3]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // Given a `rhs` tensor of shape f32[4,3]{1,0}
  // The `rhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[2,2]{1,0} slice(rhs), slice={[2:4], [1:3]}
  // rhs = [1.0, 1.0, 1.0,
  //        1.0, 1.0, 1.0,
  //        1.0, 1.0, 1.0,
  //        1.0, 1.0, 1.0]
  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(2 * 2);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  se::DeviceMemory<int64_t> rhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> rhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> rhs_offset_arr{2, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&rhs_offset_0, &rhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&rhs_offset_1, &rhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace, lhs_offset_0,
                                 lhs_offset_1, rhs_offset_0, rhs_offset_1},
                                0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Execute address computation thunk and verify that it failed because of non
  // contiguous slices on both `lhs` and `rhs`.
  ASSERT_FALSE(thunk.ExecuteOnStream(params).ok());
}

TEST(AddressComputationThunkTest, MulipleSlicedOperandsGemm) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = sizeof(float) * 2 * 4;
  int64_t out_length = sizeof(float) * 1;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(float) * 3;

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/0, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/1, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  BufferAllocation alloc_lhs(/*index=*/0, length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, length);

  BufferAllocation alloc_rhs(/*index=*/1, length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out(fake_allocations.back().get(), 0,
                                    out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace(fake_allocations.back().get(), 0,
                                          1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/4, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/5, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_rhs_offset_0(/*index=*/6, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_rhs_offset_0(&alloc_rhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_rhs_offset_1(/*index=*/7, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_rhs_offset_1(&alloc_rhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs_fake,
      slice_out, slice_workspace, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  std::vector<AddressComputationThunk::Offset> rhs_offsets{slice_rhs_offset_0,
                                                           slice_rhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, rhs_offsets, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}),
       ShapeUtil::MakeShape(PrimitiveType::F32, {8, 1}), std::nullopt,
       std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}),
       ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), std::nullopt,
       std::nullopt},
      {sizeof(int64_t), sizeof(int64_t), std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  std::vector<float> arr{1, 2, 3, 4, 5, 6, 7, 8};
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  TF_ASSERT_OK(stream->Memcpy(&lhs, arr.data(), length));

  // Given a `rhs` tensor of shape f32[8,1]{1,0}
  // The `rhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[3,1]{1,0} slice(rhs), slice={[2:5], [0:1]}
  // rhs = [1.0,
  //        2.0,
  //        3.0,
  //        4.0,
  //        5.0,
  //        6.0,
  //        7.0,
  //        8.0]
  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(8);
  std::vector<float> rhs_arr(8, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, arr.data(), length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  se::DeviceMemory<int64_t> rhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> rhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> rhs_offset_arr{2, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&rhs_offset_0, &rhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&rhs_offset_1, &rhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace, lhs_offset_0,
                                 lhs_offset_1, rhs_offset_0, rhs_offset_1},
                                0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Execute address computation thunk and verify that it executed a GEMM on the
  // right slices.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({2 * 3 + 3 * 4 + 4 * 5}));
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  return stream->MemcpyD2D(
      &dst->data, src.data,
      absl::c_accumulate(src.dimensions, 1.0, std::multiplies<int64_t>()) *
          sizeof(float));
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Ret<ffi::AnyBuffer>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", PLATFORM,
                         kMemcpy);

TEST(AddressComputationThunkTest, SlicedMemcpy) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 8;
  int64_t dst_count = 8 * 8;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * dst_count;

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/0, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_src_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/1, dst_length, /*color=*/0));
  BufferAllocation::Slice slice_dst(fake_allocations.back().get(), 0,
                                    dst_length);

  BufferAllocation alloc_offset_0(/*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_0(&alloc_offset_0, 0, offset_length);

  BufferAllocation alloc_offset_1(/*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_1(&alloc_offset_1, 0, offset_length);

  BufferAllocation alloc_offset_2(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_2(&alloc_offset_2, 0, offset_length);

  BufferAllocation alloc_offset_3(/*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_offset_3(&alloc_offset_3, 0, offset_length);

  // Preparing custom call thunk: setting up call target and operands + results
  // buffers.
  auto registration = xla::ffi::FindHandler("__xla_test$$memcpy", PLATFORM);
  ASSERT_TRUE(registration.ok());

  std::vector<std::optional<CustomCallThunk::Slice>> operands{
      CustomCallThunk::Slice{slice_src_fake,
                             ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8})}};
  std::vector<std::optional<CustomCallThunk::Slice>> results{
      CustomCallThunk::Slice{slice_dst,
                             ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<CustomCallThunk>(
      Thunk::ThunkInfo(), registration->bundle, operands, results,
      /*attributes=*/CustomCallThunk::AttributesMap(),
      /*called_computation=*/nullptr));

  // Wrapping address computation thunk around the custom call thunk.
  std::vector<AddressComputationThunk::Offset> slice_offsets{
      slice_offset_0, slice_offset_1, slice_offset_2, slice_offset_3};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_offsets, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 8}), std::nullopt},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 8, 8}), std::nullopt},
      {sizeof(int64_t), std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,8]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,8,8]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:10], [0:8]}

  // Preparing memory for thunk arguments.
  se::DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) src_arr[i] = i;
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceMemory<int64_t> offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> offset_arr{3, 5, 2, 0};
  TF_ASSERT_OK(stream->Memcpy(&offset_0, &offset_arr[0], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_1, &offset_arr[1], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_2, &offset_arr[2], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_3, &offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, offset_0, offset_1, offset_2, offset_3}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `dst` data back to host for verification.
  std::vector<int32_t> out(dst_count, 0);
  TF_ASSERT_OK(stream->Memcpy(out.data(), dst, dst_length));

  // Verifying that the right slice of `src` was copied to `dst`.
  std::vector<int32_t> ref(dst_count, 0);
  int64_t offset_val =
      offset_arr[3] +
      8 * (offset_arr[2] + 10 * (offset_arr[1] + 8 * offset_arr[0]));
  std::copy(src_arr.begin() + offset_val,
            src_arr.begin() + offset_val + dst_count, ref.begin());
  ASSERT_EQ(out, ref);
}

TEST(AddressComputationThunkTest, SlicedOutputMemcpy) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 2;
  int64_t dst_count = 2 * 2 * 2 * 2;
  int64_t slice_count = 2 * 2;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * slice_count;

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/0, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_src_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/1, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_dst_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  BufferAllocation alloc_dst(/*index=*/1, dst_length, /*color=*/0);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, dst_length);

  BufferAllocation alloc_src_offset_0(/*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_0(&alloc_src_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_1(/*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_1(&alloc_src_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_2(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_2(&alloc_src_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_3(/*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_3(&alloc_src_offset_3, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_0(/*index=*/6, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_0(&alloc_dst_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_1(/*index=*/7, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_1(&alloc_dst_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_2(/*index=*/8, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_2(&alloc_dst_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_3(/*index=*/9, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_3(&alloc_dst_offset_3, 0,
                                             offset_length);

  // Preparing custom call thunk: setting up call target and operands + results
  // buffers.
  auto registration = xla::ffi::FindHandler("__xla_test$$memcpy", PLATFORM);
  ASSERT_TRUE(registration.ok());

  std::vector<std::optional<CustomCallThunk::Slice>> operands{
      CustomCallThunk::Slice{slice_src_fake,
                             ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};
  std::vector<std::optional<CustomCallThunk::Slice>> results{
      CustomCallThunk::Slice{slice_dst_fake,
                             ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<CustomCallThunk>(
      Thunk::ThunkInfo(), registration->bundle, operands, results,
      /*attributes=*/CustomCallThunk::AttributesMap(),
      /*called_computation=*/nullptr));

  // Wrapping address computation thunk around the custom call thunk.
  std::vector<AddressComputationThunk::Offset> slice_src_offsets{
      slice_src_offset_0, slice_src_offset_1, slice_src_offset_2,
      slice_src_offset_3};
  std::vector<AddressComputationThunk::Offset> slice_dst_offsets{
      slice_dst_offset_0, slice_dst_offset_1, slice_dst_offset_2,
      slice_dst_offset_3};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_src_offsets, slice_dst_offsets},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2, 2, 2})},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2})},
      {sizeof(int64_t), sizeof(int64_t)});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,2]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:4], [0:2]}
  //
  // Given a `dst` tensor of shape s32[2,2,2,2]{3,2,1,0}
  // The `dst` slice that we want to copy into will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(dst), slice={[1:2], [1:2], [0:2], [0:2]}

  // Preparing memory for thunk arguments.
  se::DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) src_arr[i] = i;
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceMemory<int64_t> src_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> src_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> src_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> src_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> src_offset_arr{3, 5, 2, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_0, &src_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_1, &src_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_2, &src_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_3, &src_offset_arr[3], offset_length));

  se::DeviceMemory<int64_t> dst_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> dst_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> dst_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> dst_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> dst_offset_arr{1, 1, 0, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_0, &dst_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_1, &dst_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_2, &dst_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_3, &dst_offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, src_offset_0, src_offset_1, src_offset_2, src_offset_3,
       dst_offset_0, dst_offset_1, dst_offset_2, dst_offset_3},
      0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `dst` data back to host for verification.
  std::vector<int32_t> out(dst_count, 0);
  TF_ASSERT_OK(stream->Memcpy(out.data(), dst, dst_length));

  // Verifying that the right slice of `src` was copied to `dst`.
  std::vector<int32_t> ref(dst_count, 0);
  int64_t src_offset_val =
      src_offset_arr[3] +
      2 * (src_offset_arr[2] +
           10 * (src_offset_arr[1] + 8 * src_offset_arr[0]));
  int64_t dst_offset_val =
      dst_offset_arr[3] +
      2 * (dst_offset_arr[2] + 2 * (dst_offset_arr[1] + 2 * dst_offset_arr[0]));
  std::copy(src_arr.begin() + src_offset_val,
            src_arr.begin() + src_offset_val + slice_count,
            ref.begin() + dst_offset_val);
  ASSERT_EQ(out, ref);
}

TEST(AddressComputationThunkTest, SlicedGemmArbitraryArgumentOrder) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/0, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out_fake(fake_allocations.back().get(), 0,
                                         out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace_fake(fake_allocations.back().get(), 0,
                                               1024 * 1024);

  BufferAllocation alloc_lhs(/*index=*/1, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);

  BufferAllocation alloc_rhs(/*index=*/3, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);

  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);

  BufferAllocation alloc_workspace(/*index=*/0, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/4, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/5, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs_fake,
      slice_out_fake, slice_workspace_fake, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
       std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
       std::nullopt, std::nullopt},
      {sizeof(int64_t), std::nullopt, std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {workspace, lhs, out, rhs, lhs_offset_0, lhs_offset_1}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

TEST(AddressComputationThunkTest, SlicedGemmArbitraryNumberOfArguments) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/0, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out_fake(fake_allocations.back().get(), 0,
                                         out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace_fake(fake_allocations.back().get(), 0,
                                               1024 * 1024);

  BufferAllocation alloc_lhs(/*index=*/7, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);

  BufferAllocation alloc_rhs(/*index=*/3, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);

  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);

  BufferAllocation alloc_workspace(/*index=*/0, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/4, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/5, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs_fake,
      slice_out_fake, slice_workspace_fake, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
       std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
       std::nullopt, std::nullopt},
      {sizeof(int64_t), std::nullopt, std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {workspace, /*garbage, to be ignored*/ se::DeviceMemoryBase(), out, rhs,
       lhs_offset_0, lhs_offset_1, /*garbage, to be ignored*/ rhs, lhs},
      0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

TEST(AddressComputationThunkTest, SlicedTupledOperandGemm) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/0, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  BufferAllocation alloc_lhs(/*index=*/0, 3 * lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, lhs_length, lhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs(fake_allocations.back().get(), 0,
                                    rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out(fake_allocations.back().get(), 0,
                                    out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace(fake_allocations.back().get(), 0,
                                          1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/4, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/5, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs, slice_out,
      slice_workspace, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
       std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
       std::nullopt, std::nullopt},
      {sizeof(int64_t), std::nullopt, std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //

  // Preparing memory for thunk arguments.
  // lhs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  //
  // The real `lhs` tensor will look more like this:
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}
  se::DeviceMemory<float> lhs_whole_buffer =
      executor->AllocateArray<float>(2 * 4 * 3);
  TF_ASSERT_OK(stream->MemZero(&lhs_whole_buffer, 2 * 4 * 3));
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  se::DeviceMemoryBase lhs =
      lhs_whole_buffer.GetByteSlice(lhs_length, lhs_length);
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {lhs_whole_buffer, rhs, out, workspace, lhs_offset_0, lhs_offset_1}, 0,
      &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

TEST(AddressComputationThunkTest, SlicedMemcpyOOB) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 2;
  int64_t dst_count = 2 * 2 * 2 * 2;
  int64_t slice_count = 2 * 2;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * slice_count;

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/0, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_src_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/1, slice_length, /*color=*/0));
  BufferAllocation::Slice slice_dst_fake(fake_allocations.back().get(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  BufferAllocation alloc_dst(/*index=*/1, dst_length, /*color=*/0);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, dst_length);

  BufferAllocation alloc_src_offset_0(/*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_0(&alloc_src_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_1(/*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_1(&alloc_src_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_2(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_2(&alloc_src_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_src_offset_3(/*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_src_offset_3(&alloc_src_offset_3, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_0(/*index=*/6, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_0(&alloc_dst_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_1(/*index=*/7, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_1(&alloc_dst_offset_1, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_2(/*index=*/8, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_2(&alloc_dst_offset_2, 0,
                                             offset_length);

  BufferAllocation alloc_dst_offset_3(/*index=*/9, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_offset_3(&alloc_dst_offset_3, 0,
                                             offset_length);

  // Preparing custom call thunk: setting up call target and operands + results
  // buffers.
  auto registration = xla::ffi::FindHandler("__xla_test$$memcpy", PLATFORM);
  ASSERT_TRUE(registration.ok());

  std::vector<std::optional<CustomCallThunk::Slice>> operands{
      CustomCallThunk::Slice{slice_src_fake,
                             ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};
  std::vector<std::optional<CustomCallThunk::Slice>> results{
      CustomCallThunk::Slice{slice_dst_fake,
                             ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<CustomCallThunk>(
      Thunk::ThunkInfo(), registration->bundle, operands, results,
      /*attributes=*/CustomCallThunk::AttributesMap(),
      /*called_computation=*/nullptr));

  // Wrapping address computation thunk around the custom call thunk.
  std::vector<AddressComputationThunk::Offset> slice_src_offsets{
      slice_src_offset_0, slice_src_offset_1, slice_src_offset_2,
      slice_src_offset_3};
  std::vector<AddressComputationThunk::Offset> slice_dst_offsets{
      slice_dst_offset_0, slice_dst_offset_1, slice_dst_offset_2,
      slice_dst_offset_3};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_src_offsets, slice_dst_offsets},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2, 2, 2})},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2})},
      {sizeof(int64_t), sizeof(int64_t)});

  // Step 2:
  // Execute address computation thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,2]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:4], [0:2]}
  //
  // Given a `dst` tensor of shape s32[2,2,2,2]{3,2,1,0}
  // The `dst` slice that we want to copy into will be equivalent to this static
  // slice op:
  // s32[1,1,2,2]{3,2,1,0} slice(dst), slice={[1:2], [1:2], [0:2], [0:2]}

  // Preparing memory for thunk arguments.
  se::DeviceMemory<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) src_arr[i] = i;
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceMemory<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceMemory<int64_t> src_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> src_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> src_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> src_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> src_ref_offset_arr{3, 5, 2, 0};
  std::vector<int64_t> src_offset_arr{3, 5, 2, -3};
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_0, &src_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_1, &src_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_2, &src_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_3, &src_offset_arr[3], offset_length));

  se::DeviceMemory<int64_t> dst_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> dst_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> dst_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> dst_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> dst_ref_offset_arr{1, 1, 0, 0};
  std::vector<int64_t> dst_offset_arr{3, 2, 5, -4};
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_0, &dst_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_1, &dst_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_2, &dst_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&dst_offset_3, &dst_offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, src_offset_0, src_offset_1, src_offset_2, src_offset_3,
       dst_offset_0, dst_offset_1, dst_offset_2, dst_offset_3},
      0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `dst` data back to host for verification.
  std::vector<int32_t> out(dst_count, 0);
  TF_ASSERT_OK(stream->Memcpy(out.data(), dst, dst_length));

  // Verifying that the right slice of `src` was copied to `dst`.
  std::vector<int32_t> ref(dst_count, 0);
  int64_t src_offset_val =
      src_ref_offset_arr[3] +
      2 * (src_ref_offset_arr[2] +
           10 * (src_ref_offset_arr[1] + 8 * src_ref_offset_arr[0]));
  int64_t dst_offset_val =
      dst_ref_offset_arr[3] +
      2 * (dst_ref_offset_arr[2] +
           2 * (dst_ref_offset_arr[1] + 2 * dst_ref_offset_arr[0]));
  std::copy(src_arr.begin() + src_offset_val,
            src_arr.begin() + src_offset_val + slice_count,
            ref.begin() + dst_offset_val);
  ASSERT_EQ(out, ref);
}

TEST(AddressComputationThunkTest, SlicedOperandsSameBufferGemm) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Step 1:
  // Prepare embedded and address computation thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/0, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_lhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0));
  BufferAllocation::Slice slice_rhs_fake(fake_allocations.back().get(), 0,
                                         rhs_length);

  fake_allocations.push_back(
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0));
  BufferAllocation::Slice slice_out_fake(fake_allocations.back().get(), 0,
                                         out_length);

  fake_allocations.push_back(std::make_unique<BufferAllocation>(
      /*index=*/3, 1024 * 1024, /*color=*/0));
  BufferAllocation::Slice slice_workspace_fake(fake_allocations.back().get(), 0,
                                               1024 * 1024);

  BufferAllocation alloc(/*index=*/0, lhs_length + rhs_length + out_length,
                         /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc, lhs_length, rhs_length);
  BufferAllocation::Slice slice_out(&alloc, lhs_length + rhs_length,
                                    out_length);

  BufferAllocation alloc_workspace(/*index=*/1, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  BufferAllocation alloc_lhs_offset_0(/*index=*/2, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(&alloc_lhs_offset_0, 0,
                                             offset_length);

  BufferAllocation alloc_lhs_offset_1(/*index=*/3, offset_length,
                                      /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(&alloc_lhs_offset_1, 0,
                                             offset_length);

  // Preparing config for GEMM thunk.
  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config.value(), slice_lhs_fake, slice_rhs_fake,
      slice_out_fake, slice_workspace_fake, /*deterministic=*/true));

  // Wrapping address computation thunk around the GEMM thunk.
  std::vector<AddressComputationThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                           slice_lhs_offset_1};
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      {lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
       std::nullopt, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
       std::nullopt, std::nullopt},
      {sizeof(int64_t), std::nullopt, std::nullopt, std::nullopt});

  // Step 2:
  // Execute address computation thunk.
  //

  // Preparing memory for thunk arguments.
  // lhs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  //
  // The real `lhs` tensor will look more like this:
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}
  se::DeviceMemory<float> buffer =
      executor->AllocateArray<float>(lhs_length + rhs_length + out_length);
  TF_ASSERT_OK(stream->MemZero(&buffer, lhs_length + rhs_length + out_length));

  se::DeviceMemoryBase lhs = buffer.GetByteSlice(0, lhs_length);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceMemoryBase rhs = buffer.GetByteSlice(lhs_length, rhs_length);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemoryBase out =
      buffer.GetByteSlice(lhs_length + rhs_length, out_length);

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceMemory<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({buffer, workspace, lhs_offset_0, lhs_offset_1},
                                0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing address computation thunk.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

}  // namespace xla::gpu
