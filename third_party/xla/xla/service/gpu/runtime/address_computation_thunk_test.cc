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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_types.h"  // IWYU pragma: keep
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

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

  se::Stream stream(executor);
  TF_ASSERT_OK(stream.Initialize());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t lhs_offset_length = sizeof(int64_t) * 2;

  // Prepare arguments:
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream.Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream.Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream.MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream.MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset = executor->AllocateArray<int64_t>(2);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream.Memcpy(&lhs_offset, lhs_offset_arr.data(), lhs_offset_length));

  // Prepare buffer allocations and slices.
  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation alloc_lhs_offset(/*index=*/4, lhs_offset_length,
                                    /*color=*/0);

  BufferAllocation alloc_lhs_fake(/*index=*/0, rhs_length, /*color=*/0);

  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);
  BufferAllocation::Slice slice_lhs_offset(&alloc_lhs_offset, 0,
                                           lhs_offset_length);
  BufferAllocation::Slice slice_lhs_fake(&alloc_lhs_fake, 0, rhs_length);

  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Prepare embedded and address computation thunks.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(nullptr), config.value(), slice_lhs_fake, slice_rhs,
      slice_out, slice_workspace, /*deterministic=*/true));
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(nullptr),
      std::make_unique<ThunkSequence>(std::move(seq)), {slice_lhs, slice_rhs},
      {slice_out, slice_workspace}, {slice_lhs_offset, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt});

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({lhs, rhs, out, workspace, lhs_offset}, 0,
                                executor->GetAllocator());

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, &stream, &stream, {}, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, &stream, &stream}));

  // Execute address computation thunk and verify that it executed a GEMM on the
  // right slices.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream.BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream.Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

TEST(AddressComputationThunkTest, SlicedNonContiguousGemm) {
  se::StreamExecutor* executor = GpuExecutor();

  se::Stream stream(executor);
  TF_ASSERT_OK(stream.Initialize());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 2;
  int64_t offset_length = sizeof(int64_t) * 2;
  int64_t slice_length = sizeof(float) * 2 * 2;

  // Prepare arguments:
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  // rhs = [1.0, 1.0, 1.0,
  //        1.0, 1.0, 1.0,
  //        1.0, 1.0, 1.0,
  //        1.0, 1.0, 1.0]
  se::DeviceMemory<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream.Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceMemory<float> rhs = executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1);
  TF_ASSERT_OK(stream.Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<float> out = executor->AllocateArray<float>(2 * 2);
  TF_ASSERT_OK(stream.MemZero(&out, out_length));

  se::DeviceMemory<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream.MemZero(&workspace, 1024 * 1024));

  se::DeviceMemory<int64_t> lhs_offset = executor->AllocateArray<int64_t>(2);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream.Memcpy(&lhs_offset, lhs_offset_arr.data(), offset_length));

  se::DeviceMemory<int64_t> rhs_offset = executor->AllocateArray<int64_t>(2);
  std::vector<int64_t> rhs_offset_arr{2, 1};
  TF_ASSERT_OK(
      stream.Memcpy(&rhs_offset, rhs_offset_arr.data(), offset_length));

  // Prepare buffer allocations and slices.
  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/3, 1024 * 1024, /*color=*/0);

  BufferAllocation alloc_lhs_offset(/*index=*/4, offset_length, /*color=*/0);
  BufferAllocation alloc_rhs_offset(/*index=*/5, offset_length, /*color=*/0);

  BufferAllocation alloc_lhs_fake(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation alloc_rhs_fake(/*index=*/1, slice_length, /*color=*/0);

  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0, 1024 * 1024);

  BufferAllocation::Slice slice_lhs_offset(&alloc_lhs_offset, 0, offset_length);
  BufferAllocation::Slice slice_rhs_offset(&alloc_rhs_offset, 0, offset_length);

  BufferAllocation::Slice slice_lhs_fake(&alloc_lhs_fake, 0, slice_length);
  BufferAllocation::Slice slice_rhs_fake(&alloc_rhs_fake, 0, slice_length);

  auto config =
      GemmConfig::For(ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), {}, {1},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), {}, {0},
                      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}), 1.0,
                      0.0, 0.0, PrecisionConfig::ALG_UNSET, std::nullopt,
                      se::blas::kDefaultComputePrecision, false, false);
  ASSERT_TRUE(config.ok());

  // Prepare embedded and address computation thunks.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(nullptr), config.value(), slice_lhs_fake, slice_rhs_fake,
      slice_out, slice_workspace, /*deterministic=*/true));
  AddressComputationThunk thunk(
      Thunk::ThunkInfo(nullptr),
      std::make_unique<ThunkSequence>(std::move(seq)), {slice_lhs, slice_rhs},
      {slice_out, slice_workspace}, {slice_lhs_offset, slice_rhs_offset},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}),
       ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3})},
      {ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2})});

  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations(
      {lhs, rhs, out, workspace, lhs_offset, rhs_offset}, 0,
      executor->GetAllocator());

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, &stream, &stream, {}, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, &stream, &stream}));

  // Execute address computation thunk and verify that it failed because of non
  // contiguous slice.
  ASSERT_FALSE(thunk.ExecuteOnStream(params).ok());
}

}  // namespace xla::gpu
