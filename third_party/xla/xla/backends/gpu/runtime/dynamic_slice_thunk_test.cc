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

#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_multimem_registry.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.pb.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class DummyThunk : public Thunk {
 public:
  explicit DummyThunk(Kind kind, const Thunk::ThunkInfo& info)
      : Thunk(kind, info) {}
  ~DummyThunk() override = default;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

using DynamicSliceThunkTest = HloHardwareIndependentTestBase;
using ::testing::NotNull;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

std::string GetPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

se::StreamExecutor* GpuExecutor() {
  stream_executor::Platform* platform =
      se::PlatformManager::PlatformWithName(GetPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}
void CheckProtoRoundTrip(const DynamicSliceThunk& thunk,
                         const DynamicSliceThunkProto& proto) {
  std::vector<BufferAllocation> buffer_allocations;
  for (int i = 0; i < 10; ++i) {
    buffer_allocations.push_back(BufferAllocation(
        /*index=*/i, /*size=*/1024, /*color=*/0));
  }

  std::vector<BufferAllocation> fake_allocations_span;
  const auto& arguments = thunk.get_arguments();
  for (int i = 0; i < arguments.size(); ++i) {
    if (arguments[i].has_value()) {
      fake_allocations_span.push_back(
          BufferAllocation(i, arguments[i].value().allocation()->size(), 0));
    }
  }

  Thunk::DeserializerWithCustomAllocations deserializer =
      [](const ThunkProto& thunk_proto,
         absl::Span<const BufferAllocation> fake_allocations_span)
      -> absl::StatusOr<std::unique_ptr<Thunk>> {
    return DeserializeThunkProto(thunk_proto, fake_allocations_span,
                                 /*hlo_module*/ nullptr,
                                 /*platform_name=*/"TEST_PLATFORM",
                                 /*gpu_compute_capability=*/{});
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk_from_proto,
      DynamicSliceThunk::FromProto(Thunk::ThunkInfo(), proto,
                                   /*buffer_allocations=*/buffer_allocations,
                                   deserializer));
  TF_ASSERT_OK_AND_ASSIGN(auto proto_roundtrip, thunk_from_proto->ToProto());
  auto dynamic_slice_thunk_proto_roundtrip =
      proto_roundtrip.dynamic_slice_thunk();
  auto proto_no_ids = proto;
  // Hlo ids are expected to be different after roundtrip, thus we drop them
  // from comparison.
  proto_no_ids.mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_init()
      ->mutable_hlo_module()
      ->clear_id();
  proto_no_ids.mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_update()
      ->mutable_hlo_module()
      ->clear_id();
  for (auto& module_with_config :
       *proto_no_ids.mutable_offset_as_function_of_indvar_modules_metadata()
            ->mutable_extracted_offset_modules()) {
    module_with_config.mutable_hlo_module()->clear_id();
  }

  dynamic_slice_thunk_proto_roundtrip
      .mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_init()
      ->mutable_hlo_module()
      ->clear_id();
  dynamic_slice_thunk_proto_roundtrip
      .mutable_offset_as_function_of_indvar_modules_metadata()
      ->mutable_indvar_update()
      ->mutable_hlo_module()
      ->clear_id();
  for (auto& module_with_config :
       *dynamic_slice_thunk_proto_roundtrip
            .mutable_offset_as_function_of_indvar_modules_metadata()
            ->mutable_extracted_offset_modules()) {
    module_with_config.mutable_hlo_module()->clear_id();
  }

  EXPECT_THAT(dynamic_slice_thunk_proto_roundtrip, EqualsProto(proto_no_ids));
}

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>> CreateSlicedGemmThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  se::StreamExecutor* executor = GpuExecutor();
  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);
  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);

  fake_allocations.emplace_back(/*index=*/0, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);

  auto alloc_lhs =
      std::make_unique<BufferAllocation>(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc_lhs.get(), 0, lhs_length);
  fake_allocations.emplace_back(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(&fake_allocations.back(), 0, rhs_length);

  fake_allocations.emplace_back(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(&fake_allocations.back(), 0, out_length);

  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(&fake_allocations.back(), 0,
                                          1024 * 1024);

  auto alloc_lhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(alloc_lhs_offset_0.get(), 0,
                                             offset_length);

  auto alloc_lhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(alloc_lhs_offset_1.get(), 0,
                                             offset_length);

  backing_allocations.push_back(std::move(alloc_lhs));
  backing_allocations.push_back(std::move(alloc_lhs_offset_0));
  backing_allocations.push_back(std::move(alloc_lhs_offset_1));
  // Preparing config for GEMM thunk.
  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          executor->GetDeviceDescription().gpu_compute_capability()));
  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config, slice_lhs_fake, slice_rhs, slice_out,
      slice_workspace, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                     slice_lhs_offset_1};
  return std::make_unique<DynamicSliceThunk>(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<PrimitiveType>>{S64, std::nullopt, std::nullopt,
                                                std::nullopt});
}

TEST_F(DynamicSliceThunkTest, SlicedGemmProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          CreateSlicedGemmThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest, SlicedGemm) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(auto thunk,
                          CreateSlicedGemmThunk(backing_allocations));

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Execute dynamic slice thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceAddress<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceAddress<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {lhs, rhs, out, workspace, lhs_offset_0, lhs_offset_1}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>>
CreateMultipleSlicedOperandsGemmThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  se::StreamExecutor* executor = GpuExecutor();
  int64_t length = sizeof(float) * 2 * 4;
  int64_t out_length = sizeof(float) * 1;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(float) * 3;
  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(&fake_allocations.back(), 0,
                                         slice_length);
  fake_allocations.emplace_back(/*index=*/1, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs_fake(&fake_allocations.back(), 0,
                                         slice_length);
  auto alloc_lhs =
      std::make_unique<BufferAllocation>(/*index=*/0, length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc_lhs.get(), 0, length);
  auto alloc_rhs =
      std::make_unique<BufferAllocation>(/*index=*/1, length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(alloc_rhs.get(), 0, length);
  fake_allocations.emplace_back(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(&fake_allocations.back(), 0, out_length);
  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(&fake_allocations.back(), 0,
                                          1024 * 1024);
  auto alloc_lhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(alloc_lhs_offset_0.get(), 0,
                                             offset_length);
  auto alloc_lhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(alloc_lhs_offset_1.get(), 0,
                                             offset_length);
  auto alloc_rhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/6, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs_offset_0(alloc_rhs_offset_0.get(), 0,
                                             offset_length);
  auto alloc_rhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/7, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs_offset_1(alloc_rhs_offset_1.get(), 0,
                                             offset_length);

  backing_allocations.push_back(std::move(alloc_lhs));
  backing_allocations.push_back(std::move(alloc_rhs));
  backing_allocations.push_back(std::move(alloc_lhs_offset_0));
  backing_allocations.push_back(std::move(alloc_lhs_offset_1));
  backing_allocations.push_back(std::move(alloc_rhs_offset_0));
  backing_allocations.push_back(std::move(alloc_rhs_offset_1));

  // Preparing config for GEMM thunk.
  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          executor->GetDeviceDescription().gpu_compute_capability()));

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config, slice_lhs_fake, slice_rhs_fake, slice_out,
      slice_workspace, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                     slice_lhs_offset_1};
  std::vector<DynamicSliceThunk::Offset> rhs_offsets{slice_rhs_offset_0,
                                                     slice_rhs_offset_1};
  return std::make_unique<DynamicSliceThunk>(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, rhs_offsets, std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}),
          ShapeUtil::MakeShape(PrimitiveType::F32, {8, 1}), std::nullopt,
          std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}),
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), std::nullopt,
          std::nullopt},
      std::vector<std::optional<PrimitiveType>>{S64, S64, std::nullopt,
                                                std::nullopt});
}

TEST_F(DynamicSliceThunkTest, MultipleSlicedOperandsGemmProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CreateMultipleSlicedOperandsGemmThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest, MultipleSlicedOperandsGemm) {
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CreateMultipleSlicedOperandsGemmThunk(backing_allocations));

  int64_t length = sizeof(float) * 2 * 4;
  int64_t out_length = sizeof(float) * 1;
  int64_t offset_length = sizeof(int64_t);

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  std::vector<float> arr{1, 2, 3, 4, 5, 6, 7, 8};
  se::DeviceAddress<float> lhs = executor->AllocateArray<float>(2 * 4);
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
  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(8);
  TF_ASSERT_OK(stream->Memcpy(&rhs, arr.data(), length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceAddress<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  se::DeviceAddress<int64_t> rhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> rhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> rhs_offset_arr{2, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&rhs_offset_0, &rhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&rhs_offset_1, &rhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace, lhs_offset_0,
                                 lhs_offset_1, rhs_offset_0, rhs_offset_1},
                                0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Execute dynamic slice thunk and verify that it executed a GEMM on the
  // right slices.
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({2 * 3 + 3 * 4 + 4 * 5}));
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Ret<ffi::AnyBuffer>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "CUDA",
                         kMemcpy);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "ROCM",
                         kMemcpy);

TEST_F(DynamicSliceThunkTest, SlicedMemcpy) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t src_count = 8 * 8 * 10 * 8;
  int64_t dst_count = 8 * 8;
  int64_t src_length = sizeof(int32_t) * src_count;
  int64_t dst_length = sizeof(int32_t) * dst_count;
  int64_t offset_length = sizeof(int64_t);
  int64_t slice_length = sizeof(int32_t) * dst_count;

  // Step 1:
  // Prepare embedded and dynamic slice thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_src_fake(&fake_allocations.back(), 0,
                                         slice_length);

  BufferAllocation alloc_src(/*index=*/0, src_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, src_length);

  fake_allocations.emplace_back(/*index=*/1, dst_length, /*color=*/0);
  BufferAllocation::Slice slice_dst(&fake_allocations.back(), 0, dst_length);

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto registration,
      xla::ffi::FindHandler("__xla_test$$memcpy", GetPlatformName()));

  std::vector<NullableShapedSlice> operands{ShapedSlice{
      slice_src_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8})}};
  std::vector<NullableShapedSlice> results{
      ShapedSlice{slice_dst, ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  TF_ASSERT_OK_AND_ASSIGN(
      seq.emplace_back(),
      CustomCallThunk::Create(Thunk::ThunkInfo(), "__xla_test$$memcpy",
                              registration.bundle, operands, results,
                              /*attributes=*/ffi::AttributesMap(),
                              /*called_computation=*/nullptr,
                              /*gpu_compute_capability=*/{}));

  // Wrapping dynamic slice thunk around the custom call thunk.
  std::vector<DynamicSliceThunk::Offset> slice_offsets{
      slice_offset_0, slice_offset_1, slice_offset_2, slice_offset_3};
  DynamicSliceThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_offsets, std::nullopt},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 8}), std::nullopt},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 8, 8}), std::nullopt},
      {S64, std::nullopt});

  // Step 2:
  // Execute dynamic slice thunk.
  //
  // Given a `src` tensor of shape s32[8,8,10,8]{3,2,1,0}
  // The `src` slice that we want to copy from will be equivalent to this static
  // slice op:
  // s32[1,1,8,8]{3,2,1,0} slice(src), slice={[3:4], [5:6], [2:10], [0:8]}

  // Preparing memory for thunk arguments.
  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) {
    src_arr[i] = i;
  }
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceAddress<int64_t> offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> offset_arr{3, 5, 2, 0};
  TF_ASSERT_OK(stream->Memcpy(&offset_0, &offset_arr[0], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_1, &offset_arr[1], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_2, &offset_arr[2], offset_length));
  TF_ASSERT_OK(stream->Memcpy(&offset_3, &offset_arr[3], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, offset_0, offset_1, offset_2, offset_3}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
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

TEST_F(DynamicSliceThunkTest, SlicedOutputMemcpy) {
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
  // Prepare embedded and dynamic slice thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_src_fake(&fake_allocations.back(), 0,
                                         slice_length);

  fake_allocations.emplace_back(/*index=*/1, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_fake(&fake_allocations.back(), 0,
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto registration,
      xla::ffi::FindHandler("__xla_test$$memcpy", GetPlatformName()));

  std::vector<NullableShapedSlice> operands{ShapedSlice{
      slice_src_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};
  std::vector<NullableShapedSlice> results{ShapedSlice{
      slice_dst_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  TF_ASSERT_OK_AND_ASSIGN(
      seq.emplace_back(),
      CustomCallThunk::Create(Thunk::ThunkInfo(), "__xla_test$$memcpy",
                              registration.bundle, operands, results,
                              /*attributes=*/ffi::AttributesMap(),
                              /*called_computation=*/nullptr,
                              /*gpu_compute_capability=*/{}));

  // Wrapping dynamic slice thunk around the custom call thunk.
  std::vector<DynamicSliceThunk::Offset> slice_src_offsets{
      slice_src_offset_0, slice_src_offset_1, slice_src_offset_2,
      slice_src_offset_3};
  std::vector<DynamicSliceThunk::Offset> slice_dst_offsets{
      slice_dst_offset_0, slice_dst_offset_1, slice_dst_offset_2,
      slice_dst_offset_3};
  DynamicSliceThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_src_offsets, slice_dst_offsets},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2, 2, 2})},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2})},
      {S64, S64});

  // Step 2:
  // Execute dynamic slice thunk.
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
  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) {
    src_arr[i] = i;
  }
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceAddress<int64_t> src_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_3 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> src_offset_arr{3, 5, 2, 0};
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_0, &src_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_1, &src_offset_arr[1], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_2, &src_offset_arr[2], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&src_offset_3, &src_offset_arr[3], offset_length));

  se::DeviceAddress<int64_t> dst_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_3 = executor->AllocateArray<int64_t>(1);
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
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, src_offset_0, src_offset_1, src_offset_2, src_offset_3,
       dst_offset_0, dst_offset_1, dst_offset_2, dst_offset_3},
      0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
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

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>>
CreateSlicedGemmArbitraryArgumentOrderThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  se::StreamExecutor* executor = GpuExecutor();
  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(/*index=*/0, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);
  fake_allocations.emplace_back(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);
  fake_allocations.emplace_back(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out_fake(&fake_allocations.back(), 0,
                                         out_length);
  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace_fake(&fake_allocations.back(), 0,
                                               1024 * 1024);

  auto alloc_lhs =
      std::make_unique<BufferAllocation>(/*index=*/1, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc_lhs.get(), 0, lhs_length);
  auto alloc_rhs =
      std::make_unique<BufferAllocation>(/*index=*/3, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(alloc_rhs.get(), 0, rhs_length);
  auto alloc_out =
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(alloc_out.get(), 0, out_length);
  auto alloc_workspace = std::make_unique<BufferAllocation>(
      /*index=*/0, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(alloc_workspace.get(), 0,
                                          1024 * 1024);
  auto alloc_lhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(alloc_lhs_offset_0.get(), 0,
                                             offset_length);
  auto alloc_lhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(alloc_lhs_offset_1.get(), 0,
                                             offset_length);
  backing_allocations.push_back(std::move(alloc_lhs));
  backing_allocations.push_back(std::move(alloc_rhs));
  backing_allocations.push_back(std::move(alloc_out));
  backing_allocations.push_back(std::move(alloc_workspace));
  backing_allocations.push_back(std::move(alloc_lhs_offset_0));
  backing_allocations.push_back(std::move(alloc_lhs_offset_1));

  // Preparing config for GEMM thunk.
  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          executor->GetDeviceDescription().gpu_compute_capability()));

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config, slice_lhs_fake, slice_rhs_fake,
      slice_out_fake, slice_workspace_fake, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                     slice_lhs_offset_1};
  return std::make_unique<DynamicSliceThunk>(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<PrimitiveType>>{S64, std::nullopt, std::nullopt,
                                                std::nullopt});
}

TEST_F(DynamicSliceThunkTest, SlicedGemmArbitraryArgumentOrderProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CreateSlicedGemmArbitraryArgumentOrderThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest, SlicedGemmArbitraryArgumentOrder) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DynamicSliceThunk> thunk,
      CreateSlicedGemmArbitraryArgumentOrderThunk(backing_allocations));

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Execute dynamic slice thunk.
  //
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceAddress<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceAddress<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {workspace, lhs, out, rhs, lhs_offset_0, lhs_offset_1}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>>
CreateSlicedGemmArbitraryNumberOfArgumentsThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  se::StreamExecutor* executor = GpuExecutor();
  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(/*index=*/0, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);
  fake_allocations.emplace_back(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);
  fake_allocations.emplace_back(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out_fake(&fake_allocations.back(), 0,
                                         out_length);
  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace_fake(&fake_allocations.back(), 0,
                                               1024 * 1024);

  auto alloc_lhs =
      std::make_unique<BufferAllocation>(/*index=*/7, lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc_lhs.get(), 0, lhs_length);
  auto alloc_rhs =
      std::make_unique<BufferAllocation>(/*index=*/3, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(alloc_rhs.get(), 0, rhs_length);
  auto alloc_out =
      std::make_unique<BufferAllocation>(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(alloc_out.get(), 0, out_length);
  auto alloc_workspace = std::make_unique<BufferAllocation>(
      /*index=*/0, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(alloc_workspace.get(), 0,
                                          1024 * 1024);
  auto alloc_lhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(alloc_lhs_offset_0.get(), 0,
                                             offset_length);
  auto alloc_lhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(alloc_lhs_offset_1.get(), 0,
                                             offset_length);

  backing_allocations.push_back(std::move(alloc_lhs));
  backing_allocations.push_back(std::move(alloc_rhs));
  backing_allocations.push_back(std::move(alloc_out));
  backing_allocations.push_back(std::move(alloc_workspace));
  backing_allocations.push_back(std::move(alloc_lhs_offset_0));
  backing_allocations.push_back(std::move(alloc_lhs_offset_1));

  // Preparing config for GEMM thunk.
  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          executor->GetDeviceDescription().gpu_compute_capability()));

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config, slice_lhs_fake, slice_rhs_fake,
      slice_out_fake, slice_workspace_fake, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                     slice_lhs_offset_1};
  return std::make_unique<DynamicSliceThunk>(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<PrimitiveType>>{S64, std::nullopt, std::nullopt,
                                                std::nullopt});
}

TEST_F(DynamicSliceThunkTest,
       SlicedGemmArbitraryNumberOfArgumentsProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DynamicSliceThunk> thunk,
      CreateSlicedGemmArbitraryNumberOfArgumentsThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest, SlicedGemmArbitraryNumberOfArguments) {
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [1:4]}

  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DynamicSliceThunk> thunk,
      CreateSlicedGemmArbitraryNumberOfArgumentsThunk(backing_allocations));

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceAddress<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceAddress<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {workspace, /*garbage, to be ignored*/ se::DeviceAddressBase(), out, rhs,
       lhs_offset_0, lhs_offset_1, /*garbage, to be ignored*/ rhs, lhs},
      0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>>
CreateSlicedTupledOperandGemmThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  se::StreamExecutor* executor = GpuExecutor();
  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(/*index=*/0, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);

  auto alloc_lhs = std::make_unique<BufferAllocation>(
      /*index=*/0, 3 * lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc_lhs.get(), lhs_length, lhs_length);

  fake_allocations.emplace_back(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(&fake_allocations.back(), 0, rhs_length);

  fake_allocations.emplace_back(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(&fake_allocations.back(), 0, out_length);

  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(&fake_allocations.back(), 0,
                                          1024 * 1024);

  auto alloc_lhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/4, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(alloc_lhs_offset_0.get(), 0,
                                             offset_length);

  auto alloc_lhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/5, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(alloc_lhs_offset_1.get(), 0,
                                             offset_length);

  backing_allocations.push_back(std::move(alloc_lhs));
  backing_allocations.push_back(std::move(alloc_lhs_offset_0));
  backing_allocations.push_back(std::move(alloc_lhs_offset_1));

  // Preparing config for GEMM thunk.
  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          executor->GetDeviceDescription().gpu_compute_capability()));

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config, slice_lhs_fake, slice_rhs, slice_out,
      slice_workspace, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                     slice_lhs_offset_1};
  return std::make_unique<DynamicSliceThunk>(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<PrimitiveType>>{S64, std::nullopt, std::nullopt,
                                                std::nullopt});
}

TEST_F(DynamicSliceThunkTest, SlicedTupledOperandGemmProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CreateSlicedTupledOperandGemmThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest, SlicedTupledOperandGemm) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CreateSlicedTupledOperandGemmThunk(backing_allocations));

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

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
  se::DeviceAddress<float> lhs_whole_buffer =
      executor->AllocateArray<float>(2 * 4 * 3);
  TF_ASSERT_OK(stream->MemZero(&lhs_whole_buffer, 2 * 4 * 3));
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  se::DeviceAddressBase lhs =
      lhs_whole_buffer.GetByteSlice(lhs_length, lhs_length);
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(3 * 1);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(1 * 1);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceAddress<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {lhs_whole_buffer, rhs, out, workspace, lhs_offset_0, lhs_offset_1}, 0,
      &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

TEST_F(DynamicSliceThunkTest, SlicedMemcpyOOB) {
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
  // Prepare embedded and dynamic slice thunks.

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(2);

  // Fake slices for embedded thunk creation.
  fake_allocations.emplace_back(/*index=*/0, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_src_fake(&fake_allocations.back(), 0,
                                         slice_length);

  fake_allocations.emplace_back(/*index=*/1, slice_length, /*color=*/0);
  BufferAllocation::Slice slice_dst_fake(&fake_allocations.back(), 0,
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto registration,
      xla::ffi::FindHandler("__xla_test$$memcpy", GetPlatformName()));

  std::vector<NullableShapedSlice> operands{ShapedSlice{
      slice_src_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};
  std::vector<NullableShapedSlice> results{ShapedSlice{
      slice_dst_fake, ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2})}};

  // Creating embedded custom call thunk.
  ThunkSequence seq;
  TF_ASSERT_OK_AND_ASSIGN(
      seq.emplace_back(),
      CustomCallThunk::Create(Thunk::ThunkInfo(), "__xla_test$$memcpy",
                              registration.bundle, operands, results,
                              /*attributes=*/ffi::AttributesMap(),
                              /*called_computation=*/nullptr,
                              /*gpu_compute_capability=*/{}));

  // Wrapping dynamic slice thunk around the custom call thunk.
  std::vector<DynamicSliceThunk::Offset> slice_src_offsets{
      slice_src_offset_0, slice_src_offset_1, slice_src_offset_2,
      slice_src_offset_3};
  std::vector<DynamicSliceThunk::Offset> slice_dst_offsets{
      slice_dst_offset_0, slice_dst_offset_1, slice_dst_offset_2,
      slice_dst_offset_3};
  DynamicSliceThunk thunk(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      {slice_src, slice_dst}, std::move(fake_allocations),
      {slice_src_offsets, slice_dst_offsets},
      {ShapeUtil::MakeShape(PrimitiveType::S32, {8, 8, 10, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {2, 2, 2, 2})},
      // Make sure to pass a dst shape with the same rank as src shape (i.e.
      // original slice result and not bitcasted one)
      {ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2}),
       ShapeUtil::MakeShape(PrimitiveType::S32, {1, 1, 2, 2})},
      {S64, S64});

  // Step 2:
  // Execute dynamic slice thunk.
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
  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(src_count);
  std::vector<int32_t> src_arr(src_count, 0);
  for (unsigned i = 0; i < src_count; ++i) {
    src_arr[i] = i;
  }
  TF_ASSERT_OK(stream->Memcpy(&src, src_arr.data(), src_length));

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(dst_count);
  TF_ASSERT_OK(stream->MemZero(&dst, dst_length));

  se::DeviceAddress<int64_t> src_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> src_offset_3 = executor->AllocateArray<int64_t>(1);
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

  se::DeviceAddress<int64_t> dst_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_1 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_2 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> dst_offset_3 = executor->AllocateArray<int64_t>(1);
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
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(
      {src, dst, src_offset_0, src_offset_1, src_offset_2, src_offset_3,
       dst_offset_0, dst_offset_1, dst_offset_2, dst_offset_3},
      0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
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

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>>
CreateSlicedOperandsSameBufferGemmThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  se::StreamExecutor* executor = GpuExecutor();
  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(/*index=*/0, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);

  fake_allocations.emplace_back(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs_fake(&fake_allocations.back(), 0,
                                         rhs_length);

  fake_allocations.emplace_back(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_out_fake(&fake_allocations.back(), 0,
                                         out_length);

  fake_allocations.emplace_back(/*index=*/3, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace_fake(&fake_allocations.back(), 0,
                                               1024 * 1024);

  auto alloc = std::make_unique<BufferAllocation>(
      /*index=*/0, lhs_length + rhs_length + out_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc.get(), 0, lhs_length);
  BufferAllocation::Slice slice_rhs(alloc.get(), lhs_length, rhs_length);
  BufferAllocation::Slice slice_out(alloc.get(), lhs_length + rhs_length,
                                    out_length);

  auto alloc_workspace = std::make_unique<BufferAllocation>(
      /*index=*/1, 1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(alloc_workspace.get(), 0,
                                          1024 * 1024);

  auto alloc_lhs_offset_0 = std::make_unique<BufferAllocation>(
      /*index=*/2, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_0(alloc_lhs_offset_0.get(), 0,
                                             offset_length);

  auto alloc_lhs_offset_1 = std::make_unique<BufferAllocation>(
      /*index=*/3, offset_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_offset_1(alloc_lhs_offset_1.get(), 0,
                                             offset_length);

  backing_allocations.push_back(std::move(alloc));
  backing_allocations.push_back(std::move(alloc_workspace));
  backing_allocations.push_back(std::move(alloc_lhs_offset_0));
  backing_allocations.push_back(std::move(alloc_lhs_offset_1));

  // Preparing config for GEMM thunk.
  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          executor->GetDeviceDescription().gpu_compute_capability()));

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      Thunk::ThunkInfo(), config, slice_lhs_fake, slice_rhs_fake,
      slice_out_fake, slice_workspace_fake, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{slice_lhs_offset_0,
                                                     slice_lhs_offset_1};
  return std::make_unique<DynamicSliceThunk>(
      Thunk::ThunkInfo(), std::make_unique<ThunkSequence>(std::move(seq)),
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      std::move(fake_allocations),
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), std::nullopt,
          std::nullopt, std::nullopt},
      std::vector<std::optional<PrimitiveType>>{S64, std::nullopt, std::nullopt,
                                                std::nullopt});
}

TEST_F(DynamicSliceThunkTest, SlicedOperandsSameBufferGemmProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CreateSlicedOperandsSameBufferGemmThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest, SlicedOperandsSameBufferGemm) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CreateSlicedOperandsSameBufferGemmThunk(backing_allocations));

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 3 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;
  int64_t offset_length = sizeof(int64_t);

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
  se::DeviceAddress<float> buffer =
      executor->AllocateArray<float>(lhs_length + rhs_length + out_length);
  TF_ASSERT_OK(stream->MemZero(&buffer, lhs_length + rhs_length + out_length));

  se::DeviceAddressBase lhs = buffer.GetByteSlice(0, lhs_length);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  // rhs = [1.0,
  //        1.0,
  //        1.0]
  se::DeviceAddressBase rhs = buffer.GetByteSlice(lhs_length, rhs_length);
  std::vector<float> rhs_arr(3, 1);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddressBase out =
      buffer.GetByteSlice(lhs_length + rhs_length, out_length);

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(&workspace, 1024 * 1024));

  se::DeviceAddress<int64_t> lhs_offset_0 = executor->AllocateArray<int64_t>(1);
  se::DeviceAddress<int64_t> lhs_offset_1 = executor->AllocateArray<int64_t>(1);
  std::vector<int64_t> lhs_offset_arr{0, 1};
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_0, &lhs_offset_arr[0], offset_length));
  TF_ASSERT_OK(
      stream->Memcpy(&lhs_offset_1, &lhs_offset_arr[1], offset_length));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({buffer, workspace, lhs_offset_0, lhs_offset_1},
                                0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<float>({9}));
}

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>>
CreateHostInductionVariableAndOffsetEvaluationThunk(
    std::vector<std::unique_ptr<BufferAllocation>>& backing_allocations) {
  std::vector<std::unique_ptr<HloModule>> offset_modules;
  const char* offset = R"(
    HloModule offset
    ENTRY main {
      p0 = s32[] parameter(0)
      c32 = s32[] constant(32)
      c0 = s32[] constant(0)
      add = s32[] add(p0, c32)
      compare = pred[] compare(add, c0), direction=LT
      ROOT select = s32[] select(compare, add, p0)
    }
  )";
  TF_ASSIGN_OR_RETURN(auto offset_module,
                      ParseAndReturnUnverifiedModule(offset));
  offset_modules.emplace_back(std::move(offset_module));
  HloModule* offset_module_ptr = offset_modules.back().get();
  const char* indvar_init = R"(
    HloModule indvar_init
    ENTRY main {
      ROOT c0 = s32[] constant(0)
    }
  )";
  TF_ASSIGN_OR_RETURN(auto indvar_init_module,
                      ParseAndReturnUnverifiedModule(indvar_init));
  const char* indvar_update = R"(
    HloModule indvar_update
    ENTRY main {
      p0 = s32[] parameter(0)
      c1 = s32[] constant(1)
      ROOT add = s32[] add(p0, c1)
    }
  )";
  TF_ASSIGN_OR_RETURN(auto indvar_update_module,
                      ParseAndReturnUnverifiedModule(indvar_update));
  se::StreamExecutor* executor = GpuExecutor();

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;

  // Preparing buffer allocation slices for thunk creations.
  std::vector<BufferAllocation> fake_allocations;
  fake_allocations.reserve(4);
  fake_allocations.emplace_back(/*index=*/0, /*size=*/rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs_fake(
      /*allocation=*/&fake_allocations.back(), /*offset=*/0,
      /*size=*/rhs_length);

  auto alloc_lhs = std::make_unique<BufferAllocation>(
      /*index=*/0, /*size=*/lhs_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(alloc_lhs.get(), /*offset=*/0,
                                    /*size=*/lhs_length);

  fake_allocations.emplace_back(/*index=*/1, /*size=*/rhs_length, /*color=*/0);
  BufferAllocation::Slice slice_rhs(
      /*allocation=*/&fake_allocations.back(), /*offset=*/0,
      /*size=*/rhs_length);

  fake_allocations.emplace_back(/*index=*/2, /*size=*/out_length, /*color=*/0);
  BufferAllocation::Slice slice_out(
      /*allocation=*/&fake_allocations.back(), /*offset=*/0,
      /*size=*/out_length);

  fake_allocations.emplace_back(/*index=*/3, /*size=*/1024 * 1024, /*color=*/0);
  BufferAllocation::Slice slice_workspace(
      /*allocation=*/&fake_allocations.back(), /*offset=*/0,
      /*size=*/1024 * 1024);

  backing_allocations.push_back(std::move(alloc_lhs));

  // Preparing config for GEMM thunk.

  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(
          /*lhs_shape=*/ShapeUtil::MakeShape(
              /*element_type=*/PrimitiveType::F32,
              /*dimensions=*/{1, 4}),
          /*lhs_batch_dims=*/{}, /*lhs_contracting_dims=*/{1},
          /*rhs_shape=*/
          ShapeUtil::MakeShape(/*element_type=*/PrimitiveType::F32,
                               /*dimensions=*/{4, 1}),
          /*rhs_batch_dims=*/{}, /*rhs_contracting_dims=*/{0},
          /*output_shape=*/
          ShapeUtil::MakeShape(/*element_type=*/PrimitiveType::F32,
                               /*dimensions=*/{1, 1}),
          /*alpha_real=*/1.0, /*alpha_imag=*/0.0, /*beta=*/0.0,
          /*precision_algorithm=*/PrecisionConfig::ALG_UNSET,
          /*algorithm=*/std::nullopt,
          /*compute_precision=*/se::blas::kDefaultComputePrecision,
          /*grad_x=*/false, /*grad_y=*/false,
          /*gpu_version=*/
          executor->GetDeviceDescription().gpu_compute_capability()));

  // Creating embedded GEMM thunk.
  ThunkSequence seq;
  seq.emplace_back(std::make_unique<GemmThunk>(
      /*thunk_info*/ Thunk::ThunkInfo(), /*config=*/config,
      /*lhs_buffer=*/slice_lhs_fake, /*rhs_buffer=*/slice_rhs,
      /*output_buffer=*/slice_out,
      /*workspace=*/slice_workspace, /*deterministic=*/true));

  // Wrapping dynamic slice thunk around the GEMM thunk.
  std::vector<DynamicSliceThunk::Offset> lhs_offsets{offset_module_ptr, 0l};
  DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata
      offset_as_function_of_indvar_modules_metadata(
          std::move(indvar_init_module), std::move(indvar_update_module),
          std::move(offset_modules));
  return std::make_unique<DynamicSliceThunk>(
      /*thunk_info=*/Thunk::ThunkInfo(),
      /*embedded_thunk=*/std::make_unique<ThunkSequence>(std::move(seq)),
      /*arguments=*/
      std::vector<std::optional<BufferAllocation::Slice>>{
          slice_lhs, slice_rhs, slice_out, slice_workspace},
      /*fake_allocations=*/std::move(fake_allocations),
      /*offsets=*/
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>{
          lhs_offsets, std::nullopt, std::nullopt, std::nullopt},
      /*orig_shapes=*/
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      /*sliced_shapes=*/
      std::vector<std::optional<Shape>>{
          ShapeUtil::MakeShape(PrimitiveType::F32, {1, 4}), std::nullopt,
          std::nullopt, std::nullopt},
      /*offset_byte_sizes=*/
      std::vector<std::optional<PrimitiveType>>{S64, std::nullopt, std::nullopt,
                                                std::nullopt},
      /*offset_as_function_of_indvar_metadata=*/
      std::move(offset_as_function_of_indvar_modules_metadata));
}

TEST_F(
    DynamicSliceThunkTest,
    HostInductionVariableAndOffsetEvaluationExecutesCorrectlyProtoRoundTrip) {
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CreateHostInductionVariableAndOffsetEvaluationThunk(backing_allocations));
  TF_ASSERT_OK_AND_ASSIGN(auto proto, thunk->ToProto());
  CheckProtoRoundTrip(*thunk, proto.dynamic_slice_thunk());
}

TEST_F(DynamicSliceThunkTest,
       HostInductionVariableAndOffsetEvaluationExecutesCorrectly) {
  // Given a `lhs` tensor of shape f32[2,4]{1,0}
  // The `lhs` slice that we want to use will be equivalent to this static
  // slice op:
  // f32[1,3]{1,0} slice(lhs), slice={[0:1], [0:4]}

  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  std::vector<std::unique_ptr<BufferAllocation>> backing_allocations;
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CreateHostInductionVariableAndOffsetEvaluationThunk(backing_allocations));

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 1;
  int64_t out_length = sizeof(float) * 1 * 1;

  // Preparing memory for thunk arguments.
  // lhs = [1.0, 2.0, 3.0, 4.0,
  //        5.0, 6.0, 7.0, 8.0]
  se::DeviceAddress<float> lhs =
      executor->AllocateArray<float>(/*element_count=*/2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(/*gpu_dst=*/&lhs, /*host_src=*/lhs_arr.data(),
                              /*size=*/lhs_length));

  // rhs = [4.0,
  //        3.0,
  //        2.0,
  //        1.0]
  se::DeviceAddress<float> rhs =
      executor->AllocateArray<float>(/*element_count=*/4 * 1);
  std::vector<float> rhs_arr{4, 3, 2, 1};
  TF_ASSERT_OK(stream->Memcpy(/*gpu_dst=*/&rhs, /*host_src=*/rhs_arr.data(),
                              /*size=*/rhs_length));

  se::DeviceAddress<float> out =
      executor->AllocateArray<float>(/*element_count=*/1 * 1);
  TF_ASSERT_OK(stream->MemZero(/*location=*/&out, /*size=*/out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(/*element_count=*/1024 * 1024);
  TF_ASSERT_OK(stream->MemZero(/*location=*/&workspace, /*size=*/1024 * 1024));

  // Preparing parameters for thunk execution.
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations(/*buffers=*/{lhs, rhs, out, workspace},
                                /*device_ordinal=*/executor->device_ordinal(),
                                /*memory_allocator=*/&allocator);

  CollectiveCliqueRequests clique_requests;
  CollectiveMemoryRequests memory_requests(allocations);
  CollectiveMultimemRegistry multimem_registry(
      executor, collective_params.global_device_id);

  Thunk::PrepareParams prepare_params{&collective_params, &clique_requests,
                                      &memory_requests,   &multimem_registry,
                                      executor,           &allocations};

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, /*buffer_allocations=*/allocations, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk->Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  // Executing dynamic slice thunk.
  TF_ASSERT_OK(thunk->Prepare(prepare_params));
  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  std::vector<float> dst(1, 0);
  TF_ASSERT_OK(stream->Memcpy(/*host_dst=*/dst.data(), /*gpu_src=*/out,
                              /*size=*/out_length));

  ASSERT_EQ(dst, std::vector<float>({1 * 4 + 2 * 3 + 3 * 2 + 4 * 1}));

  TF_ASSERT_OK(thunk->ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copying `out` data back to host for verification.
  TF_ASSERT_OK(stream->Memcpy(/*host_dst=*/dst.data(), /*gpu_src=*/out,
                              /*size=*/out_length));

  EXPECT_EQ(dst, std::vector<float>({5 * 4 + 6 * 3 + 7 * 2 + 8 * 1}));
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithNullopt) {
  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::nullopt;
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto,
      SerializeOptionalDynamicSliceOffsetsToProto(offsets_item, std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_offsets,
      DeserializeOptionalDynamicSliceOffsetsFromProto(proto, {}, std::nullopt));
  EXPECT_FALSE(deserialized_offsets.has_value());
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithConstOffset) {
  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::vector<DynamicSliceThunk::Offset>{123l};
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto,
      SerializeOptionalDynamicSliceOffsetsToProto(offsets_item, std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_offsets,
      DeserializeOptionalDynamicSliceOffsetsFromProto(proto, {}, std::nullopt));
  ASSERT_TRUE(deserialized_offsets.has_value());
  ASSERT_EQ(deserialized_offsets->size(), 1);
  EXPECT_EQ(std::get<int64_t>((*deserialized_offsets)[0]), 123l);
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithSliceOffset) {
  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(0, 1024, 0);
  BufferAllocation::Slice slice(&allocations.back(), 128, 256);
  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::vector<DynamicSliceThunk::Offset>{slice};
  TF_ASSERT_OK_AND_ASSIGN(
      auto proto,
      SerializeOptionalDynamicSliceOffsetsToProto(offsets_item, std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized_offsets,
                          DeserializeOptionalDynamicSliceOffsetsFromProto(
                              proto, allocations, std::nullopt));
  ASSERT_TRUE(deserialized_offsets.has_value());
  ASSERT_EQ(deserialized_offsets->size(), 1);
  auto deserialized_slice =
      std::get<BufferAllocation::Slice>((*deserialized_offsets)[0]);
  EXPECT_EQ(deserialized_slice.allocation(), &allocations.back());
  EXPECT_EQ(deserialized_slice.offset(), 128);
  EXPECT_EQ(deserialized_slice.size(), 256);
}

TEST_F(DynamicSliceThunkTest,
       SerializeAndDeserializeOptionalOffsetsWithHloModuleOffset) {
  const char* hlo_text = R"(
      HloModule test_module
      ENTRY main {
        ROOT c = f32[] constant(0.0)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(hlo_text));
  HloModule* hlo_module_ptr = hlo_module.get();

  std::vector<std::unique_ptr<HloModule>> modules;
  modules.push_back(std::move(hlo_module));

  const char* indvar_init_hlo = R"(
      HloModule indvar_init
      ENTRY main {
        ROOT c0 = s32[] constant(0)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_init_module,
                          ParseAndReturnUnverifiedModule(indvar_init_hlo));

  const char* indvar_update_hlo = R"(
      HloModule indvar_update
      ENTRY main {
        p0 = s32[] parameter(0)
        c1 = s32[] constant(1)
        ROOT add = s32[] add(p0, c1)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_update_module,
                          ParseAndReturnUnverifiedModule(indvar_update_hlo));

  DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata metadata(
      std::move(indvar_init_module), std::move(indvar_update_module),
      std::move(modules));

  std::optional<std::vector<DynamicSliceThunk::Offset>> offsets_item =
      std::vector<DynamicSliceThunk::Offset>{hlo_module_ptr};

  TF_ASSERT_OK_AND_ASSIGN(auto proto,
                          SerializeOptionalDynamicSliceOffsetsToProto(
                              offsets_item, std::move(metadata)));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module2,
                          ParseAndReturnUnverifiedModule(hlo_text));
  std::vector<std::unique_ptr<HloModule>> modules2;
  modules2.push_back(std::move(hlo_module2));
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_init_module2,
                          ParseAndReturnUnverifiedModule(indvar_init_hlo));
  TF_ASSERT_OK_AND_ASSIGN(auto indvar_update_module2,
                          ParseAndReturnUnverifiedModule(indvar_update_hlo));
  DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata metadata2(
      std::move(indvar_init_module2), std::move(indvar_update_module2),
      std::move(modules2));
  TF_ASSERT_OK_AND_ASSIGN(auto deserialized_offsets,
                          DeserializeOptionalDynamicSliceOffsetsFromProto(
                              proto, {}, std::move(metadata2)));
  ASSERT_TRUE(deserialized_offsets.has_value());
  ASSERT_EQ(deserialized_offsets->size(), 1);
  EXPECT_TRUE(std::holds_alternative<HloModule*>((*deserialized_offsets)[0]));
  EXPECT_NE(std::get<HloModule*>((*deserialized_offsets)[0]), nullptr);
  EXPECT_EQ(proto.offsets().offsets(0).hlo_module_offset_idx(), 0);
}

TEST_F(DynamicSliceThunkTest, TransformAllNestedThunks) {
  auto seq = std::make_unique<ThunkSequence>();
  seq->emplace_back(
      std::make_unique<DummyThunk>(Thunk::Kind::kGemm, Thunk::ThunkInfo()));
  DynamicSliceThunk thunk(Thunk::ThunkInfo(),
                          /*embedded_thunk=*/std::move(seq),
                          /*arguments=*/{},
                          /*fake_allocations=*/{},
                          /*offsets=*/{},
                          /*orig_shapes=*/{},
                          /*sliced_shapes=*/{},
                          /*offset_byte_sizes=*/{});

  TF_EXPECT_OK(thunk.TransformAllNestedThunks([](auto) {
    return std::make_unique<DummyThunk>(Thunk::Kind::kCustomCall,
                                        Thunk::ThunkInfo());
  }));

  EXPECT_THAT(thunk.get_embedded_thunk(), NotNull());
  EXPECT_THAT(thunk.get_embedded_thunk()->thunks(), SizeIs(1));
  EXPECT_THAT(thunk.get_embedded_thunk()->thunks()[0]->kind(),
              Thunk::Kind::kCustomCall);
}

}  // namespace
}  // namespace xla::gpu
