/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/comparison_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::tsl::proto_testing::EqualsProto;

using Parameter = DynamicSliceFusion::Parameter;
using Result = DynamicSliceFusion::Result;
using Offset = DynamicSliceFusion::Offset;

static absl::StatusOr<se::StreamExecutor*> CreateExecutor() {
  ASSIGN_OR_RETURN(std::string platform_name,
                   PlatformUtil::CanonicalPlatformName("gpu"));
  ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName(platform_name));
  return platform->ExecutorForDevice(0);
}

DynamicSliceConfig MakeConfig(int64_t loop_index, int64_t offset,
                              int64_t stride) {
  DynamicSliceConfig config;
  config.set_loop_index(loop_index);
  config.set_byte_offset(offset);
  config.set_byte_stride(stride);
  return config;
}

// A fake thunk that records the buffer addresses it receives during execution.
class BufferOffsetRecordingThunk : public Thunk {
 public:
  explicit BufferOffsetRecordingThunk(int num_buffers)
      : Thunk(Kind::kKernel, Thunk::ThunkInfo()), num_buffers_(num_buffers) {}

  absl::Status Prepare(const PrepareParams&) override {
    return absl::OkStatus();
  }

  absl::Status Initialize(const InitializeParams&) override {
    return absl::OkStatus();
  }

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    recorded_buffers_.clear();
    for (int i = 0; i < num_buffers_; ++i) {
      recorded_buffers_.push_back(
          params.buffer_allocations->GetDeviceAddress(i));
    }
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override { return {}; }

  absl::StatusOr<ThunkProto> ToProto() const override {
    return absl::UnimplementedError("not serializable");
  }

  const std::vector<se::DeviceAddressBase>& recorded_buffers() const {
    return recorded_buffers_;
  }

 private:
  int num_buffers_;
  std::vector<se::DeviceAddressBase> recorded_buffers_;
};

Thunk::ExecuteParams MakeExecuteParams(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  static ServiceExecutableRunOptions run_options;
  return Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream,
      /*command_buffer_trace_stream=*/nullptr, /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);
}

TEST(DynamicSliceFusionV2ThunkTest, VerifyBufferAssignment) {
  BufferAllocation parameter_buffer(0, 1024, 0);
  BufferAllocation unaliased_result_buffer(1, 1024, 0);
  Shape result_shape = ShapeUtil::MakeShape(F32, {256});
  Shape update_shape = ShapeUtil::MakeShape(F32, {64});

  std::vector<Result> results = {
      {0, 0, result_shape, update_shape, MakeConfig(0, 0, 256)}};

  std::vector<BufferAllocation::Slice> parameter_buffers = {
      BufferAllocation::Slice(&parameter_buffer, 0, 1024)};
  std::vector<BufferAllocation::Slice> aliased_result_buffers = {
      BufferAllocation::Slice(&parameter_buffer, 0, 1024)};
  std::vector<BufferAllocation::Slice> unaliased_result_buffers = {
      BufferAllocation::Slice(&unaliased_result_buffer, 0, 1024)};

  EXPECT_OK(DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      results, parameter_buffers, aliased_result_buffers));

  absl::Status status = DynamicSliceFusionV2Thunk::VerifyBufferAssignment(
      results, parameter_buffers, unaliased_result_buffers);
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("must alias")));
}

//===----------------------------------------------------------------------===//
// Basic tests
//===----------------------------------------------------------------------===//

TEST(DynamicSliceFusionV2ThunkTest, NoSlicing) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 1024> buf0;
  std::array<char, 1024> buf1;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf0.data(), buf0.size()),
      se::DeviceAddressBase(buf1.data(), buf1.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer0(0, 1024, 0);
  BufferAllocation buffer1(1, 1024, 0);

  std::vector<BufferAllocation> slice_allocs = {
      BufferAllocation(0, 1024, 0),
      BufferAllocation(1, 1024, 0),
  };

  Shape shape = ShapeUtil::MakeShape(F32, {256});
  std::vector<Parameter> parameters = {{0, shape, shape}};
  std::vector<Result> results = {{std::nullopt, 0, shape, shape}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(2);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, results,
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer0, 0, 1024)},
      /*result_buffers=*/{BufferAllocation::Slice(&buffer1, 0, 1024)},
      slice_allocs, ThunkSequence::Of(std::move(recording)));

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 2);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf0.data());
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].opaque(), buf1.data());
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].size(), 1024);
}

TEST(DynamicSliceFusionV2ThunkTest, StaticOffset) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 4096> buf;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf.data(), buf.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer(0, 4096, 0);
  std::vector<BufferAllocation> slice_allocs = {BufferAllocation(0, 1024, 0)};

  Shape param_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {256});
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(0, 512, 0)}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(1);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, /*results=*/{},
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer, 0, 4096)},
      /*result_buffers=*/{}, slice_allocs,
      ThunkSequence::Of(std::move(recording)));

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 1);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf.data() + 512);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
}

//===----------------------------------------------------------------------===//
// Loop-dependent tests
//===----------------------------------------------------------------------===//

TEST(DynamicSliceFusionV2ThunkTest, LoopDependentOffset) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 8192> buf;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf.data(), buf.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer(0, 8192, 0);
  std::vector<BufferAllocation> slice_allocs = {BufferAllocation(0, 1024, 0)};

  Shape param_shape = ShapeUtil::MakeShape(F32, {2048});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {256});
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(0, 0, 1024)}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(1);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, /*results=*/{},
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer, 0, 8192)},
      /*result_buffers=*/{}, slice_allocs,
      ThunkSequence::Of(std::move(recording)));

  ScopedWhileLoop loop("test_loop", 8);
  loop.IncLoopIteration();  // 1
  loop.IncLoopIteration();  // 2
  loop.IncLoopIteration();  // 3

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // offset = 0 + 3 * 1024 = 3072.
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 1);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf.data() + 3072);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
}

TEST(DynamicSliceFusionV2ThunkTest, LoopDependentWithBaseOffset) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 8192> buf;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf.data(), buf.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer(0, 8192, 0);
  std::vector<BufferAllocation> slice_allocs = {BufferAllocation(0, 1024, 0)};

  Shape param_shape = ShapeUtil::MakeShape(F32, {2048});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {256});
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(0, 256, 1024)}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(1);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, /*results=*/{},
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer, 0, 8192)},
      /*result_buffers=*/{}, slice_allocs,
      ThunkSequence::Of(std::move(recording)));

  ScopedWhileLoop loop("test_loop", 8);
  loop.IncLoopIteration();  // 1
  loop.IncLoopIteration();  // 2

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // offset = 256 + 2 * 1024 = 2304.
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 1);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf.data() + 2304);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
}

TEST(DynamicSliceFusionV2ThunkTest, VerifiesComputedOffsetExpression) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 64> buf;
  se::DeviceAddress<int32_t> offset = executor->AllocateArray<int32_t>(1, 0);
  int32_t offset_value = 1;
  ASSERT_TRUE(stream->Memcpy(&offset, &offset_value, sizeof(int32_t)).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf.data(), buf.size()), offset};
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer(0, buf.size(), 0);
  BufferAllocation offset_buffer(1, sizeof(int32_t), 0);
  std::vector<BufferAllocation> slice_allocs = {BufferAllocation(0, 16, 0)};

  Shape param_shape = ShapeUtil::MakeShape(S32, {4, 4});
  Shape slice_shape = ShapeUtil::MakeShape(S32, {1, 4});
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(0, 32, 0),
       std::vector<Offset>{
           Offset{0, Offset::Add(Offset::Parameter(1), Offset::Constant(1))},
           Offset{1, Offset::Constant(0)}}}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(1);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, /*results=*/{},
      /*parameter_buffers=*/
      {BufferAllocation::Slice(&buffer, 0, buf.size()),
       BufferAllocation::Slice(&offset_buffer, 0, sizeof(int32_t))},
      /*result_buffers=*/{}, slice_allocs,
      ThunkSequence::Of(std::move(recording)), /*verify_offsets=*/true);

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 1);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf.data() + 32);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 16);
}

//===----------------------------------------------------------------------===//
// Argument and result slicing
//===----------------------------------------------------------------------===//

TEST(DynamicSliceFusionV2ThunkTest, ArgumentAndResultSlicing) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 4096> buf0;
  std::array<char, 4096> buf1;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf0.data(), buf0.size()),
      se::DeviceAddressBase(buf1.data(), buf1.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer0(0, 4096, 0);
  BufferAllocation buffer1(1, 4096, 0);
  std::vector<BufferAllocation> slice_allocs = {
      BufferAllocation(0, 1024, 0),
      BufferAllocation(1, 1024, 0),
  };

  Shape param_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {256});
  Shape result_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape update_shape = ShapeUtil::MakeShape(F32, {256});
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(0, 0, 1024)}};
  std::vector<Result> results = {
      {1, 0, result_shape, update_shape, MakeConfig(0, 0, 1024)}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(2);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, results,
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer0, 0, 4096)},
      /*result_buffers=*/{BufferAllocation::Slice(&buffer1, 0, 4096)},
      slice_allocs, ThunkSequence::Of(std::move(recording)));

  ScopedWhileLoop loop("test_loop", 4);
  loop.IncLoopIteration();  // 1

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // iteration=1, stride=1024 => offset=1024.
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 2);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf0.data() + 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].opaque(), buf1.data() + 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].size(), 1024);
}

//===----------------------------------------------------------------------===//
// Nested loops
//===----------------------------------------------------------------------===//

TEST(DynamicSliceFusionV2ThunkTest, NestedLoops) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 16384> buf;
  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf.data(), buf.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer(0, 16384, 0);
  std::vector<BufferAllocation> slice_allocs = {BufferAllocation(0, 4096, 0)};

  Shape param_shape = ShapeUtil::MakeShape(F32, {4096});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {1024});
  // Depends on outer loop (loop_index=1).
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(1, 0, 4096)}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(1);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, /*results=*/{},
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer, 0, 16384)},
      /*result_buffers=*/{}, slice_allocs,
      ThunkSequence::Of(std::move(recording)));

  // Outer loop at iteration 2, inner loop at iteration 5.
  ScopedWhileLoop outer("outer_loop", 4);
  outer.IncLoopIteration();  // 1
  outer.IncLoopIteration();  // 2

  ScopedWhileLoop inner("inner_loop", 8);
  inner.IncLoopIteration();  // 1
  inner.IncLoopIteration();  // 2
  inner.IncLoopIteration();  // 3
  inner.IncLoopIteration();  // 4
  inner.IncLoopIteration();  // 5

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // loop_nest = [outer, inner]. loop_index=1 => nest[size-1-1] = nest[0] =
  // outer at iteration 2. offset = 0 + 2 * 4096 = 8192.
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 1);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf.data() + 8192);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 4096);
}

TEST(DynamicSliceFusionV2ThunkTest, NotInsideLoop) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 4096> buf;
  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf.data(), buf.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer(0, 4096, 0);
  std::vector<BufferAllocation> slice_allocs = {BufferAllocation(0, 1024, 0)};

  Shape param_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape slice_shape = ShapeUtil::MakeShape(F32, {256});
  std::vector<Parameter> parameters = {
      {0, param_shape, slice_shape, MakeConfig(0, 128, 1024)}};

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(1);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, /*results=*/{},
      /*parameter_buffers=*/{BufferAllocation::Slice(&buffer, 0, 4096)},
      /*result_buffers=*/{}, slice_allocs,
      ThunkSequence::Of(std::move(recording)));

  // No ScopedWhileLoop — not inside any loop.
  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // iteration=0, offset = 128 + 0*1024 = 128.
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 1);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf.data() + 128);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
}

//===----------------------------------------------------------------------===//
// Multi-output tests
//===----------------------------------------------------------------------===//

TEST(DynamicSliceFusionV2ThunkTest, TwoResultsDifferentStrides) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 4096> buf0;
  std::array<char, 4096> buf1;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf0.data(), buf0.size()),
      se::DeviceAddressBase(buf1.data(), buf1.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer0(0, 4096, 0);
  BufferAllocation buffer1(1, 4096, 0);
  std::vector<BufferAllocation> slice_allocs = {
      BufferAllocation(0, 1024, 0),
      BufferAllocation(1, 1024, 0),
  };

  Shape result_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape update_shape = ShapeUtil::MakeShape(F32, {256});
  // Result 0: forward, result 1: backward.
  std::vector<Result> results = {
      {0, 0, result_shape, update_shape, MakeConfig(0, 0, 1024)},
      {1, 1, result_shape, update_shape, MakeConfig(0, 3072, -1024)},
  };

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(2);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), /*parameters=*/{}, results,
      /*parameter_buffers=*/{},
      /*result_buffers=*/
      {BufferAllocation::Slice(&buffer0, 0, 4096),
       BufferAllocation::Slice(&buffer1, 0, 4096)},
      slice_allocs, ThunkSequence::Of(std::move(recording)));

  ScopedWhileLoop loop("test_loop", 4);
  loop.IncLoopIteration();  // 1
  loop.IncLoopIteration();  // 2

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // iteration=2.
  // Result 0: 0 + 2*1024 = 2048 (forward).
  // Result 1: 3072 + 2*(-1024) = 1024 (backward).
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 2);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf0.data() + 2048);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].opaque(), buf1.data() + 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].size(), 1024);
}

TEST(DynamicSliceFusionV2ThunkTest, OneSlicedOnePassthrough) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  std::array<char, 4096> buf0;
  std::array<char, 1024> buf1;

  std::vector<se::DeviceAddressBase> addrs = {
      se::DeviceAddressBase(buf0.data(), buf0.size()),
      se::DeviceAddressBase(buf1.data(), buf1.size()),
  };
  BufferAllocations allocs(addrs, 0, nullptr);

  BufferAllocation buffer0(0, 4096, 0);
  BufferAllocation buffer1(1, 1024, 0);
  std::vector<BufferAllocation> slice_allocs = {
      BufferAllocation(0, 1024, 0),
      BufferAllocation(1, 1024, 0),
  };

  Shape result_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape update_shape = ShapeUtil::MakeShape(F32, {256});
  Shape passthrough_shape = ShapeUtil::MakeShape(F32, {256});
  // Result 0: sliced with stride. Result 1: passthrough (no slicing).
  std::vector<Result> results = {
      {0, 0, result_shape, update_shape, MakeConfig(0, 0, 1024)},
      {std::nullopt, 1, passthrough_shape, passthrough_shape},
  };

  auto recording = std::make_unique<BufferOffsetRecordingThunk>(2);
  BufferOffsetRecordingThunk* recording_ptr = recording.get();

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), /*parameters=*/{}, results,
      /*parameter_buffers=*/{},
      /*result_buffers=*/
      {BufferAllocation::Slice(&buffer0, 0, 4096),
       BufferAllocation::Slice(&buffer1, 0, 1024)},
      slice_allocs, ThunkSequence::Of(std::move(recording)));

  ScopedWhileLoop loop("test_loop", 4);
  loop.IncLoopIteration();  // 1
  loop.IncLoopIteration();  // 2
  loop.IncLoopIteration();  // 3

  auto params = MakeExecuteParams(allocs, stream.get());
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());

  // iteration=3.
  // Result 0: 0 + 3*1024 = 3072 (sliced).
  // Result 1: not sliced, full buffer at buf1.
  ASSERT_EQ(recording_ptr->recorded_buffers().size(), 2);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].opaque(), buf0.data() + 3072);
  EXPECT_EQ(recording_ptr->recorded_buffers()[0].size(), 1024);
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].opaque(), buf1.data());
  EXPECT_EQ(recording_ptr->recorded_buffers()[1].size(), 1024);
}

TEST(DynamicSliceFusionV2ThunkTest,
     CommandBufferUpdatesLoopDependentSliceOffset) {
  ASSERT_OK_AND_ASSIGN(auto* executor, CreateExecutor());
  if (executor->GetDeviceDescription().gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "DynamicSliceFusionV2Thunk command buffer updates are not "
                    "supported on ROCm";
  }
  if (executor->GetDeviceDescription().runtime_version() <
      stream_executor::SemanticVersion(12, 9, 0)) {
    GTEST_SKIP() << "DynamicSliceFusionV2Thunk command buffer updates require "
                    "CUDA runtime version 12.9 or higher";
  }
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  constexpr int64_t kSrcBytes = sizeof(int32_t) * 4;
  constexpr int64_t kSliceBytes = sizeof(int32_t);

  std::vector<BufferAllocation> embedded_allocations;
  embedded_allocations.reserve(2);
  embedded_allocations.emplace_back(/*index=*/0, kSliceBytes, /*color=*/0);
  BufferAllocation::Slice embedded_src(&embedded_allocations.back(), 0,
                                       kSliceBytes);
  embedded_allocations.emplace_back(/*index=*/1, kSliceBytes, /*color=*/0);
  BufferAllocation::Slice embedded_dst(&embedded_allocations.back(), 0,
                                       kSliceBytes);

  Shape src_shape = ShapeUtil::MakeShape(S32, {4});
  Shape slice_shape = ShapeUtil::MakeShape(S32, {1});

  ThunkSequence embedded_thunks;
  embedded_thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{embedded_src, slice_shape},
      ShapedSlice{embedded_dst, slice_shape}, kSliceBytes));

  CommandSequence embedded_commands;
  embedded_commands.Emplace<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{embedded_src, slice_shape},
      ShapedSlice{embedded_dst, slice_shape}, kSliceBytes);
  ASSERT_OK_AND_ASSIGN(CommandExecutor embedded_executor,
                       CommandExecutor::Create(
                           std::move(embedded_commands),
                           CommandExecutor::SynchronizationMode::kSerialize));

  BufferAllocation src_alloc(0, kSrcBytes, 0);
  BufferAllocation dst_alloc(1, kSliceBytes, 0);

  auto dynamic_slice_thunk = std::make_unique<DynamicSliceFusionV2Thunk>(
      Thunk::ThunkInfo(),
      std::vector<Parameter>{
          {0, src_shape, slice_shape, MakeConfig(0, 0, kSliceBytes)}},
      std::vector<Result>{{std::nullopt, 0, slice_shape, slice_shape}},
      std::vector<BufferAllocation::Slice>{
          BufferAllocation::Slice(&src_alloc, 0, kSrcBytes)},
      std::vector<BufferAllocation::Slice>{
          BufferAllocation::Slice(&dst_alloc, 0, kSliceBytes)},
      std::move(embedded_allocations), std::move(embedded_thunks));
  ASSERT_OK(dynamic_slice_thunk->SetOrUpdateCommandBufferExecutor(
      std::move(embedded_executor)));

  CommandSequence commands;
  commands.Append(dynamic_slice_thunk.get());
  ASSERT_OK_AND_ASSIGN(CommandExecutor command_executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kSerialize));

  CommandBufferThunk command_buffer_thunk(
      std::move(command_executor), Thunk::ThunkInfo(),
      /*thunks=*/nullptr,
      /*enable_command_buffers_during_profiling=*/true,
      DebugOptions::ALWAYS_UPDATE);

  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(4, 0);
  std::vector<int32_t> src_data{10, 20, 30, 40};
  ASSERT_TRUE(stream->Memcpy(&src, src_data.data(), kSrcBytes).ok());

  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(1, 0);
  ASSERT_TRUE(stream->MemZero(&dst, kSliceBytes).ok());

  stream_executor::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({src, dst}, executor->device_ordinal(),
                                &allocator);

  Thunk::PrepareParams prepare_params;
  prepare_params.executor = executor;
  prepare_params.buffer_allocations = &allocations;
  ASSERT_OK(command_buffer_thunk.Prepare(prepare_params));

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  ASSERT_OK(command_buffer_thunk.Initialize(
      {executor, source, &allocations, stream.get(), stream.get()}));

  auto params = MakeExecuteParams(allocations, stream.get());

  ScopedWhileLoop loop("dynamic_slice_fusion_v2_command_buffer",
                       /*trip_count=*/4);
  ASSERT_OK(command_buffer_thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<int32_t> out(1, 0);
  ASSERT_TRUE(stream->Memcpy(out.data(), dst, kSliceBytes).ok());
  ASSERT_EQ(out, std::vector<int32_t>({10}));

  loop.IncLoopIteration();
  ASSERT_OK(command_buffer_thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  ASSERT_TRUE(stream->Memcpy(out.data(), dst, kSliceBytes).ok());
  ASSERT_EQ(out, std::vector<int32_t>({20}));
}

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

TEST(DynamicSliceFusionV2ThunkTest, SerializeDeserializeRoundTrip) {
  BufferAllocation buffer0(0, 4096, 0);
  BufferAllocation buffer1(1, 4096, 0);

  std::vector<BufferAllocation> slice_allocs = {
      BufferAllocation(0, 1024, 0),
      BufferAllocation(1, 512, 0),
      BufferAllocation(2, 1024, 0),
  };

  Shape arg_shape = ShapeUtil::MakeShape(F32, {256});
  Shape arg2_shape = ShapeUtil::MakeShape(F32, {128});
  Shape res_shape = ShapeUtil::MakeShape(F32, {128});

  std::vector<Parameter> parameters = {
      {0, arg_shape, arg_shape, MakeConfig(0, 0, 1024),
       std::vector<Offset>{
           Offset{0, Offset::Add(Offset::Constant(1), Offset::Constant(2))}}},
      {1, arg2_shape, arg2_shape},
  };

  std::vector<Result> results = {
      {1, 0, res_shape, res_shape, MakeConfig(0, 256, 512),
       std::vector<Offset>{Offset{
           0, Offset::Select(
                  Offset::Compare(ComparisonDirection::kLt,
                                  Offset::Parameter(0), Offset::Constant(4)),
                  Offset::Parameter(0), Offset::Constant(4))}}},
  };

  ShapedSlice memzero_dest = {
      BufferAllocation::Slice(&slice_allocs[2], 0, 1024), res_shape};

  DynamicSliceFusionV2Thunk thunk(
      Thunk::ThunkInfo(), parameters, results,
      /*parameter_buffers=*/
      {BufferAllocation::Slice(&buffer0, 0, 4096),
       BufferAllocation::Slice(&buffer1, 0, 4096)},
      /*result_buffers=*/{BufferAllocation::Slice(&buffer1, 0, 4096)},
      slice_allocs,
      ThunkSequence::Of(
          std::make_unique<MemzeroThunk>(Thunk::ThunkInfo(), memzero_dest)));

  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  const auto& dsf = proto.dynamic_slice_fusion_thunk();

  std::vector<BufferAllocation> orig_allocs = {buffer0, buffer1};

  auto deserializer = [](const ThunkProto& thunk_proto,
                         absl::Span<const BufferAllocation> allocs) {
    Thunk::ThunkInfo info;
    info.profile_annotation = thunk_proto.thunk_info().profile_annotation();
    return MemzeroThunk::FromProto(std::move(info), thunk_proto.memzero_thunk(),
                                   allocs);
  };

  ASSERT_OK_AND_ASSIGN(auto deserialized,
                       DynamicSliceFusionV2Thunk::FromProto(
                           Thunk::ThunkInfo(), dsf, orig_allocs, deserializer));

  EXPECT_EQ(deserialized->buffer_uses().size(), 3);
  EXPECT_EQ(deserialized->kind(), Thunk::Kind::kDynamicSliceFusion);
  EXPECT_EQ(deserialized->thunks().size(), 1);
  EXPECT_EQ(deserialized->thunks()[0]->kind(), Thunk::Kind::kMemzero);

  // Verify deserialized parameters and results match the originals.
  EXPECT_THAT(deserialized->parameters(),
              testing::ElementsAreArray(parameters));
  EXPECT_THAT(deserialized->results(), testing::ElementsAreArray(results));

  // Re-serialize and verify the roundtrip is lossless.
  ASSERT_OK_AND_ASSIGN(ThunkProto roundtrip_proto, deserialized->ToProto());
  EXPECT_THAT(roundtrip_proto.dynamic_slice_fusion_thunk(), EqualsProto(dsf));
}

}  // namespace
}  // namespace xla::gpu
