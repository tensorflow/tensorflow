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

#include "xla/service/cpu/runtime/kernel_thunk.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

class AddF32HostKernels : public Thunk::HostKernels {
 public:
  absl::StatusOr<SE_HOST_Kernel*> Find(std::string_view name) override {
    return +[](const SE_HOST_KernelCallFrame* call_frame) {
      SE_HOST_KernelArg& in = call_frame->args[0];
      SE_HOST_KernelArg& out = call_frame->args[1];

      float* in_ptr = reinterpret_cast<float*>(in.data);
      float* out_ptr = reinterpret_cast<float*>(out.data);

      uint64_t i = call_frame->thread->x;
      *(out_ptr + i) = *(in_ptr + i) + *(in_ptr + i);

      return static_cast<SE_HOST_KernelError*>(nullptr);
    };
  }
};

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
  std::vector<BufferAllocation::Slice> slices = {in_slice, out_slice};

  KernelThunk thunk({"add_f32"}, slices, "add_f32", se::ThreadDim(4));

  AddF32HostKernels host_kernels;
  Thunk::ExecuteParams params = {&host_kernels, &allocations};
  TF_ASSERT_OK(thunk.Execute(params));

  std::vector<float> expected = {2.0, 4.0, 6.0, 8.0};
  EXPECT_EQ(out, expected);
}

}  // namespace
}  // namespace xla::cpu
