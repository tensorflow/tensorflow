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

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

class AddF32HostKernel : public FunctionLibrary {
 public:
  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        std::string_view name) final {
    auto kernel = +[](const XLA_CPU_KernelCallFrame* call_frame) {
      const XLA_CPU_KernelArg& in = call_frame->args[0];
      const XLA_CPU_KernelArg& out = call_frame->args[1];

      float* in_ptr = reinterpret_cast<float*>(in.data);
      float* out_ptr = reinterpret_cast<float*>(out.data);

      uint64_t i = call_frame->thread->x;
      *(out_ptr + i) = *(in_ptr + i) + *(in_ptr + i);

      return static_cast<XLA_CPU_KernelError*>(nullptr);
    };
    return reinterpret_cast<void*>(kernel);
  }
};

TEST(KernelThunkTest, CheckAlignment) {
  auto thunk =
      KernelThunk::Create({"test"}, {}, {}, "test", se::ThreadDim(), {},
                          /*min_alignment=*/3);
  EXPECT_TRUE(absl::StrContains(thunk.status().message(),
                                "minimum alignment 3 is not a power of 2"));
}

TEST(KernelThunkTest, AddF32) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> in = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> out(4, 0.0);

  size_t size_in_bytes = in.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(in.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation in_alloc(0, size_in_bytes, 0);
  BufferAllocation out_alloc(1, size_in_bytes, 0);

  BufferAllocation::Slice in_slice(&in_alloc, 0, size_in_bytes);
  BufferAllocation::Slice out_slice(&out_alloc, 0, size_in_bytes);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      KernelThunk::Create({"add_f32"}, {in_slice}, {out_slice}, "add_f32",
                          se::ThreadDim(4), /*invariant_arguments=*/{0}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  std::vector<float> expected = {2.0, 4.0, 6.0, 8.0};
  EXPECT_EQ(out, expected);
}

TEST(KernelThunkTest, AddF32Inline) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> in_out = {1.0, 2.0, 3.0, 4.0};

  size_t size_in_bytes = in_out.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(in_out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);
  BufferAllocation in_out_alloc(0, size_in_bytes, 0);
  BufferAllocation::Slice in_out_slice(&in_out_alloc, 0, size_in_bytes);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, KernelThunk::Create(
                      {"add_f32"}, {in_out_slice}, {in_out_slice}, "add_f32",
                      se::ThreadDim(4), /*invariant_arguments=*/{}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  std::vector<float> expected = {2.0, 4.0, 6.0, 8.0};
  EXPECT_EQ(in_out, expected);
}

TEST(KernelThunkInvariantBuffersTest, MissingBufferSlice) {
#ifdef NDEBUG
  GTEST_SKIP() << "Invariant buffers check is disabled in optimized build.";
#endif

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> in = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> out(4, 0.0);

  size_t size_in_bytes = in.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(in.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation in_alloc(0, size_in_bytes, 0);
  BufferAllocation out_alloc(1, size_in_bytes, 0);

  BufferAllocation::Slice in_slice(&in_alloc, 0, size_in_bytes);
  BufferAllocation::Slice out_slice(&out_alloc, 0, size_in_bytes);

  // Invariant buffer set is incorrect - should include in_slice, but is empty.
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      KernelThunk::Create({"add_f32"}, {in_slice}, {out_slice}, "add_f32",
                          se::ThreadDim(4), /*invariant_arguments=*/{}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsError());

  auto status = execute_event.GetError();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Mismatch in invariant buffers metadata"));
}

TEST(KernelThunkInvariantBuffersTest, ExtraInputOutputBufferSlice) {
#ifdef NDEBUG
  GTEST_SKIP() << "Invariant buffers check is disabled in optimized build.";
#endif

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> in_out = {1.0, 2.0, 3.0, 4.0};

  size_t size_in_bytes = in_out.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(in_out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);
  BufferAllocation in_out_alloc(0, size_in_bytes, 0);
  BufferAllocation::Slice in_out_slice(&in_out_alloc, 0, size_in_bytes);

  // Invariant buffer set is incorrect - should be empty, but contains input
  // buffer that's not invariant.
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, KernelThunk::Create(
                      {"add_f32"}, {in_out_slice}, {in_out_slice}, "add_f32",
                      se::ThreadDim(4), /*invariant_arguments=*/{0}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsError());

  auto status = execute_event.GetError();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Mismatch in invariant buffers metadata"));
}

// This case should never happen in practice, it simulates a bug in the code
// that incorrectly sets up aliases.
TEST(KernelThunkInvariantBuffersTest,
     MemorySectionIncorrectlyMarkedAsInvariant) {
#ifdef NDEBUG
  GTEST_SKIP() << "Invariant buffers check is disabled in optimized build.";
#endif

  // We've got only one memory section
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> in_out = {1.0, 2.0, 3.0, 4.0};

  // We've got two buffer slices with different indexes, but both pointing to
  // the same memory section.
  size_t size_in_bytes = in_out.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(in_out.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(in_out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation in_0_alloc(0, size_in_bytes, 0);
  BufferAllocation in_1_alloc(1, size_in_bytes, 0);

  BufferAllocation::Slice in_0_slice(&in_0_alloc, 0, size_in_bytes);
  BufferAllocation::Slice in_1_slice(&in_1_alloc, 0, size_in_bytes);

  // Invariant buffer set is incorrect. in_1_slice is not aliased to any output,
  // but it points to the same memory section as in_0_slice (which is not
  // invariant, because is aliased with the output).
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, KernelThunk::Create({"add_f32"}, {in_0_slice, in_1_slice},
                                      {in_0_slice}, "add_f32", se::ThreadDim(4),
                                      /*invariant_arguments=*/{1}));

  AddF32HostKernel host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsError());

  auto status = execute_event.GetError();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Mismatch in invariant buffers metadata"));
}

}  // namespace
}  // namespace xla::cpu
