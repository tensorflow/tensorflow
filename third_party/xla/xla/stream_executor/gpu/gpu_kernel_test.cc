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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

using AddI32Kernel =
    TypedKernelFactory<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                       DeviceMemory<int32_t>>;

class GpuKernelTest : public ::testing::Test {
 public:
  void SetUp() override {
    auto name = absl::AsciiStrToUpper(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    Platform* platform = PlatformManager::PlatformWithName(name).value();
    executor_ = platform->ExecutorForDevice(0).value();
  }

  void RunAddI32Kernel(const MultiKernelLoaderSpec& spec) {
    TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
    TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor_, spec));

    int64_t length = 4;
    int64_t byte_length = sizeof(int32_t) * length;

    // Prepare arguments: a=1, b=2, c=0
    DeviceMemory<int32_t> a = executor_->AllocateArray<int32_t>(length, 0);
    DeviceMemory<int32_t> b = executor_->AllocateArray<int32_t>(length, 0);
    DeviceMemory<int32_t> c = executor_->AllocateArray<int32_t>(length, 0);

    TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
    TF_ASSERT_OK(stream->Memset32(&b, 2, byte_length));
    TF_ASSERT_OK(stream->MemZero(&c, byte_length));

    // Launch kernel.
    ASSERT_TRUE(
        add.Launch(ThreadDim(), BlockDim(4), stream.get(), a, b, c).ok());

    // Copy data back to host.
    std::vector<int32_t> dst(4, 42);
    TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

    std::vector<int32_t> expected = {3, 3, 3, 3};
    ASSERT_EQ(dst, expected);
  }

  StreamExecutor* executor_;
};

TEST_F(GpuKernelTest, LoadAndRunKernelFromPtx) {
  if (executor_->GetPlatform()->id() ==
      stream_executor::rocm::kROCmPlatformId) {
    GTEST_SKIP() << "There is no PTX or any equivalent abstraction for ROCm.";
  }

  RunAddI32Kernel(GetAddI32PtxKernelSpec());
}

TEST_F(GpuKernelTest, LoadAndRunKernelFromCubin) {
  MultiKernelLoaderSpec spec(/*arity=*/3);
  TF_ASSERT_OK_AND_ASSIGN(auto binary, GetGpuTestKernelsFatbin());
  spec.AddCudaCubinInMemory(binary, "AddI32");
  RunAddI32Kernel(spec);
}

TEST_F(GpuKernelTest, LoadAndRunKernelFromSymbol) {
  RunAddI32Kernel(GetAddI32KernelSpec());
}

TEST_F(GpuKernelTest, ArrayArgByValue) {
  constexpr absl::string_view copy_kernel = R"(
    .version 8.0
    .target sm_60
    .address_size 64

    .visible .entry copy_kernel(
        .param .u64 foo_param_0,
        .param .align 1 .b8 foo_param_1[16]
)
{
        .reg .b16       %rs<17>;
        .reg .b64       %rd<3>;
        .loc    1 5 0

        ld.param.u64    %rd1, [foo_param_0];
        cvta.to.global.u64      %rd2, %rd1;
        ld.param.u8     %rs1, [foo_param_1+15];
        ld.param.u8     %rs2, [foo_param_1+14];
        ld.param.u8     %rs3, [foo_param_1+13];
        ld.param.u8     %rs4, [foo_param_1+12];
        ld.param.u8     %rs5, [foo_param_1+11];
        ld.param.u8     %rs6, [foo_param_1+10];
        ld.param.u8     %rs7, [foo_param_1+9];
        ld.param.u8     %rs8, [foo_param_1+8];
        ld.param.u8     %rs9, [foo_param_1+7];
        ld.param.u8     %rs10, [foo_param_1+6];
        ld.param.u8     %rs11, [foo_param_1+5];
        ld.param.u8     %rs12, [foo_param_1+4];
        ld.param.u8     %rs13, [foo_param_1+3];
        ld.param.u8     %rs14, [foo_param_1+2];
        ld.param.u8     %rs15, [foo_param_1+1];
        ld.param.u8     %rs16, [foo_param_1];
        .loc    1 6 5
        st.global.u8    [%rd2], %rs16;
        st.global.u8    [%rd2+1], %rs15;
        st.global.u8    [%rd2+2], %rs14;
        st.global.u8    [%rd2+3], %rs13;
        st.global.u8    [%rd2+4], %rs12;
        st.global.u8    [%rd2+5], %rs11;
        st.global.u8    [%rd2+6], %rs10;
        st.global.u8    [%rd2+7], %rs9;
        st.global.u8    [%rd2+8], %rs8;
        st.global.u8    [%rd2+9], %rs7;
        st.global.u8    [%rd2+10], %rs6;
        st.global.u8    [%rd2+11], %rs5;
        st.global.u8    [%rd2+12], %rs4;
        st.global.u8    [%rd2+13], %rs3;
        st.global.u8    [%rd2+14], %rs2;
        st.global.u8    [%rd2+15], %rs1;
        .loc    1 7 1
        ret;
    }
    )";

  MultiKernelLoaderSpec spec(/*arity=*/2);
  spec.AddCudaPtxInMemory(copy_kernel, "copy_kernel");

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto kernel, executor_->LoadKernel(spec));

  constexpr int64_t kLength = 16;

  DeviceMemory<char> dst = executor_->AllocateArray<char>(kLength, 0);
  TF_ASSERT_OK(stream->MemZero(&dst, kLength));

  struct ByValArg {
    std::byte storage[16];
  };
  ByValArg arg;
  int i = 0;
  for (auto& element : arg.storage) {
    element = static_cast<std::byte>(i++);
  }

  // Launch kernel.
  auto args = stream_executor::PackKernelArgs(/*shmem_bytes=*/0, dst, arg);
  TF_ASSERT_OK(kernel->Launch(ThreadDim(), BlockDim(), stream.get(), *args));

  // Copy data back to host.
  std::byte dst_host[16] = {};
  TF_ASSERT_OK(stream->Memcpy(dst_host, dst, kLength));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_THAT(dst_host, ::testing::ElementsAreArray(arg.storage));
}
}  // namespace
}  // namespace stream_executor::gpu
