/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/kernel_thunk.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

class AddF32HostKernel : public FunctionLibrary {
 public:
  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        absl::string_view name) final {
    auto kernel = +[](const XLA_CPU_KernelCallFrame* call_frame) {
      const XLA_CPU_KernelArg& in = call_frame->args[0];
      const XLA_CPU_KernelArg& out = call_frame->args[1];

      float* in_ptr = reinterpret_cast<float*>(in.data);
      float* out_ptr = reinterpret_cast<float*>(out.data);

      uint64_t i = call_frame->workgroup_id->x;
      *(out_ptr + i) = *(in_ptr + i) + *(in_ptr + i);

      return static_cast<XLA_CPU_KernelError*>(nullptr);
    };
    return reinterpret_cast<void*>(kernel);
  }
};

TEST(KernelThunkTest, CheckAlignment) {
  auto thunk =
      KernelThunk::Create({"test"}, {}, {}, "test", NumWorkGroups{1, 1, 1}, {},
                          /*min_alignment=*/3);
  EXPECT_TRUE(absl::StrContains(thunk.status().message(),
                                "minimum alignment 3 is not a power of 2"));
}

TEST(KernelThunkTest, AddF32) {
  auto in = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto out = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(in, out);

  auto [in_alloc, out_alloc] = CreateBufferAllocation(in, out);
  auto [in_slice, out_slice] = CreateBufferAllocationSlice(in_alloc, out_alloc);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      KernelThunk::Create({"add_f32"}, {in_slice}, {out_slice}, "add_f32",
                          NumWorkGroups{4}, /*invariant_arguments=*/{0}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  EXPECT_EQ(out, LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}}));
}

TEST(KernelThunkTest, AddF32Inline) {
  auto in_out = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  BufferAllocations allocations = CreateBufferAllocations(in_out);

  BufferAllocation alloc = CreateBufferAllocation(0, in_out);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      KernelThunk::Create({"add_f32"}, {slice}, {slice}, "add_f32",
                          NumWorkGroups{4}, /*invariant_arguments=*/{}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(in_out, LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}}));
}

TEST(KernelThunkInvariantBuffersTest, MissingBufferSlice) {
#ifdef NDEBUG
  GTEST_SKIP() << "Invariant buffers check is disabled in optimized build.";
#endif

  auto in = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto out = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(in, out);

  auto [in_alloc, out_alloc] = CreateBufferAllocation(in, out);
  auto [in_slice, out_slice] = CreateBufferAllocationSlice(in_alloc, out_alloc);

  // Invariant buffer set is incorrect - should include in_slice, but is empty.
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      KernelThunk::Create({"add_f32"}, {in_slice}, {out_slice}, "add_f32",
                          NumWorkGroups{4}, /*invariant_arguments=*/{}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsError());

  auto status = execute_event.GetError();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Argument not marked as invariant but doesn't alias with any "
      "results"));
}

TEST(KernelThunkInvariantBuffersTest, ExtraInputOutputBufferSlice) {
#ifdef NDEBUG
  GTEST_SKIP() << "Invariant buffers check is disabled in optimized build.";
#endif

  auto in_out = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  BufferAllocations allocations = CreateBufferAllocations(in_out);

  BufferAllocation alloc = CreateBufferAllocation(0, in_out);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  // Invariant buffer set is incorrect - should be empty, but contains input
  // buffer that's not invariant.
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      KernelThunk::Create({"add_f32"}, {slice}, {slice}, "add_f32",
                          NumWorkGroups{4}, /*invariant_arguments=*/{0}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsError());

  auto status = execute_event.GetError();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(
      status.message(), "Argument marked as invariant aliases with a result"));
}

// This case should never happen in practice, it simulates a bug in the code
// that incorrectly sets up aliases.
TEST(KernelThunkInvariantBuffersTest,
     MemorySectionIncorrectlyMarkedAsInvariant) {
#ifdef NDEBUG
  GTEST_SKIP() << "Invariant buffers check is disabled in optimized build.";
#endif

  // Thunk is correctly configured to have two arguments and the second marked
  // as invariant.
  auto data0 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto data1 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  auto [alloc_0, alloc_1] = CreateBufferAllocation(data0, data1);
  auto [slice_0, slice_1] = CreateBufferAllocationSlice(alloc_0, alloc_1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, KernelThunk::Create({"add_f32"}, {slice_0, slice_1},
                                      {slice_0}, "add_f32", NumWorkGroups{4},
                                      /*invariant_arguments=*/{1}));

  AddF32HostKernel host_kernels;

  // But runtime output buffer overlaps with invariant input buffer.
  std::array<float, 5> runtime_buffer;
  BufferAllocations runtime_allocations(BufferAllocations::Buffers{
      se::DeviceMemoryBase(runtime_buffer.data(), 16),
      se::DeviceMemoryBase(runtime_buffer.data() + 1, 16)});
  Thunk::ExecuteParams params = {&host_kernels, &runtime_allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsError());

  auto status = execute_event.GetError();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(
      status.message(), "Argument marked as invariant aliases with a result"));
}

}  // namespace
}  // namespace xla::cpu
