/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace se = stream_executor;

static const char *dyn_shmem_ptx = R"(
.version 4.2
.target sm_30
.address_size 64

.extern .shared .align 4 .b8 s[];

.visible .entry dyn_shmem_kernel(.param .u64 buf) {
.reg .b32 %r<6>;
.reg .b64 %rd<9>;

ld.param.u64 %rd1, [buf];
cvta.to.global.u64 %rd2, %rd1;
mov.u32 %r1, %tid.x;
mov.u32 %r2, 511;
sub.s32 %r3, %r2, %r1;
mul.wide.s32 %rd3, %r1, 4;
add.s64 %rd4, %rd2, %rd3;
ld.global.u32 %r4, [%rd4];
mov.u64 %rd5, s;
add.s64 %rd6, %rd5, %rd3;
st.shared.u32 [%rd6], %r4;
bar.sync 0;
mul.wide.s32 %rd7, %r3, 4;
add.s64 %rd8, %rd5, %rd7;
ld.shared.u32 %r5, [%rd8];
st.global.u32 [%rd4], %r5;
ret;
}
)";

TEST(ShmemTest, ReverseArray) {
  // Testing that dynamic shared memory is allocated to kernels requesting it.
  se::Platform *platform =
      se::MultiPlatformManager::PlatformWithName("cuda").value();
  se::StreamExecutor *executor = platform->ExecutorForDevice(0).value();
  se::Stream stream(executor);
  stream.Init();

  constexpr int n_elements = 512;
  using dtype = int32_t;
  constexpr int buffer_size_bytes = n_elements * sizeof(dtype);

  se::DeviceMemory<dtype> dev_buf = executor->AllocateArray<dtype>(n_elements);
  std::vector<dtype> host_buf(n_elements);
  for (int i = 0; i < n_elements; ++i) {
    host_buf[i] = i;
  }
  stream.ThenMemcpy(&dev_buf, host_buf.data(), buffer_size_bytes);
  TF_CHECK_OK(stream.BlockHostUntilDone());

  std::vector<uint8_t> compiled_ptx =
      se::CompileGpuAsm(executor->device_ordinal(), dyn_shmem_ptx,
                        PtxOptsFromDebugOptions(DebugOptions{}))
          .value();

  auto kernel = CreateKernel("dyn_shmem_kernel", /*num_args=*/1,
                             reinterpret_cast<char *>(compiled_ptx.data()),
                             /*cubin_data=*/{}, executor,
                             /*shared_mem_bytes=*/n_elements * sizeof(dtype))
                    .value();
  ExecuteKernelOnStream(
      *kernel, {dev_buf},
      {/*block_x_count=*/1, /*thread_x_count_per_block=*/n_elements}, &stream)
      .ok();
  TF_CHECK_OK(stream.BlockHostUntilDone());
  stream.ThenMemcpy(host_buf.data(), dev_buf, n_elements * sizeof(dtype));
  for (int i = 0; i < n_elements; ++i) {
    EXPECT_EQ(host_buf[i], n_elements - 1 - i);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
