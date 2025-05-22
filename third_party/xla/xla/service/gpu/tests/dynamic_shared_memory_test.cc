/* Copyright 2022 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace se = stream_executor;

// PTX below is compiled from the following CUDA code:
//
// __global__ void dyn_shmem_kernel(dtype* buf, uint32_t* n_cols,
//                                  uint32_t* n_rows) {
//   extern __shared__ dtype shmem[];
//   uint32_t src_col = threadIdx.x;
//   uint32_t dst_col = *n_cols - src_col - 1;
//   for (uint32_t i = 0; i < *n_rows; i++) {
//     shmem[src_col + *n_cols * i] = buf[src_col + *n_cols * i];
//   }
//   __syncthreads();
//   for (uint32_t i = 0; i < *n_rows; i++) {
//     buf[dst_col + *n_cols * (*n_rows - i - 1)] =
//         shmem[src_col + *n_cols * i];
//   }
// }

const absl::string_view kPTX = R"(
.version 4.2
.target sm_30
.address_size 64

.extern .shared .align 1 .b8 shmem[];

.visible .entry dyn_shmem_kernel(
.param .u64 buf,
.param .u64 n_cols,
.param .u64 n_rows
)
{
.reg .pred %p<5>;
.reg .b16 %rs<3>;
.reg .b32 %r<30>;
.reg .b64 %rd<17>;

ld.param.u64 %rd4, [buf];
ld.param.u64 %rd5, [n_rows];
cvta.to.global.u64 %rd1, %rd5;
ld.param.u64 %rd6, [n_cols];
cvta.to.global.u64 %rd2, %rd6;
cvta.to.global.u64 %rd3, %rd4;
mov.u32 %r1, %tid.x;
ld.global.u32 %r12, [%rd2];
ld.global.u32 %r14, [%rd1];
setp.eq.s32 %p1, %r14, 0;
mov.u64 %rd16, shmem;
@%p1 bra $L__BB0_3;
mov.u32 %r26, 0;
$L__BB0_2:
ld.global.u32 %r16, [%rd2];
mad.lo.s32 %r17, %r16, %r26, %r1;
cvt.u64.u32 %rd7, %r17;
add.s64 %rd8, %rd3, %rd7;
ld.global.u8 %rs1, [%rd8];
add.s64 %rd10, %rd16, %rd7;
st.shared.u8 [%rd10], %rs1;
add.s32 %r26, %r26, 1;
ld.global.u32 %r18, [%rd1];
setp.lt.u32 %p2, %r26, %r18;
@%p2 bra $L__BB0_2;
$L__BB0_3:
bar.sync 0;
ld.global.u32 %r28, [%rd1];
setp.eq.s32 %p3, %r28, 0;
@%p3 bra $L__BB0_6;
not.b32 %r13, %r1;
add.s32 %r2, %r12, %r13;
mov.u32 %r29, 0;
mov.u32 %r27, -1;
$L__BB0_5:
ld.global.u32 %r21, [%rd2];
mad.lo.s32 %r22, %r21, %r29, %r1;
cvt.u64.u32 %rd11, %r22;
add.s64 %rd13, %rd16, %rd11;
ld.shared.u8 %rs2, [%rd13];
add.s32 %r23, %r28, %r27;
mad.lo.s32 %r24, %r21, %r23, %r2;
cvt.u64.u32 %rd14, %r24;
add.s64 %rd15, %rd3, %rd14;
st.global.u8 [%rd15], %rs2;
add.s32 %r29, %r29, 1;
ld.global.u32 %r28, [%rd1];
add.s32 %r27, %r27, -1;
setp.gt.u32 %p4, %r28, %r29;
@%p4 bra $L__BB0_5;
$L__BB0_6:
ret;
})";

TEST(SharedMemoryUseTest, ArrayReversalWorks) {
  // Test that shared memory is fully available to kernels requesting it.
  // Create an array with a 2D pattern of numbers, fill the requested shared
  // memory with it, read it back inverting both axes,
  // copy the result back to the host and verify it.
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("cuda").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Use 90% of the available shared memory to verify that a fractional
  // amount works as well, not only the full size.
  const unsigned n_cols =
      executor->GetDeviceDescription().threads_per_block_limit();
  const unsigned n_rows =
      0.9 * executor->GetDeviceDescription().shared_memory_per_block_optin() /
      n_cols;
  const int n_elements = n_cols * n_rows;
  using data_type = uint8_t;
  constexpr int max_value = UINT8_MAX;
  const int buffer_size_bytes = n_elements * sizeof(data_type);
  VLOG(1) << "Using " << buffer_size_bytes << " bytes of shared memory";

  std::unique_ptr<stream_executor::Kernel> kernel =
      CreateKernel("dyn_shmem_kernel", /*num_args=*/3, kPTX,
                   /*cubin_data=*/{}, executor,
                   /*shared_mem_bytes=*/buffer_size_bytes)
          .value();

  se::DeviceMemory<data_type> device_buffer =
      executor->AllocateArray<data_type>(n_elements);
  std::vector<data_type> host_buffer(n_elements);
  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      // Fill the buffer with a reasonably non-uniform pattern, multiples of
      // 3 and 5 make it non-symmetric with respect to the main diagonal.
      host_buffer[row * n_cols + col] = (3 * col + 5 * row) % max_value;
    }
  }

  TF_CHECK_OK(
      stream->Memcpy(&device_buffer, host_buffer.data(), buffer_size_bytes));
  se::DeviceMemory<uint32_t> dev_n_cols = executor->AllocateScalar<uint32_t>();
  TF_CHECK_OK(stream->Memcpy(&dev_n_cols, &n_cols, sizeof(uint32_t)));
  se::DeviceMemory<uint32_t> dev_n_rows = executor->AllocateScalar<uint32_t>();
  TF_CHECK_OK(stream->Memcpy(&dev_n_rows, &n_rows, sizeof(uint32_t)));
  TF_CHECK_OK(stream->BlockHostUntilDone());

  TF_CHECK_OK(ExecuteKernelOnStream(
      *kernel, {device_buffer, dev_n_cols, dev_n_rows},
      {/*block_x_count=*/1, /*thread_x_count_per_block=*/n_cols},
      /*cluster_dim=*/{}, stream.get()));
  TF_CHECK_OK(stream->BlockHostUntilDone());
  TF_CHECK_OK(
      stream->Memcpy(host_buffer.data(), device_buffer, buffer_size_bytes));
  TF_CHECK_OK(stream->BlockHostUntilDone());

  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      EXPECT_EQ(host_buffer[(n_rows - row - 1) * n_cols + (n_cols - col - 1)],
                (3 * col + 5 * row) % max_value)
          << row << " " << col;
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
