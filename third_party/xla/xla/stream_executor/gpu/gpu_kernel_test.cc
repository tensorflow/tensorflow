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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/gpu/tma_metadata.pb.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace stream_executor::gpu {
namespace {
using ::testing::Each;
using tsl::proto_testing::ParseTextProtoOrDie;

using AddI32Kernel =
    TypedKernelFactory<DeviceAddress<int32_t>, DeviceAddress<int32_t>,
                       DeviceAddress<int32_t>>;
using TmaKernel = TypedKernelFactory<TensorMap, TensorMap, TensorMap>;

class GpuKernelTest : public ::testing::Test {
 public:
  void SetUp() override {
    auto name = absl::AsciiStrToUpper(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    Platform* platform = PlatformManager::PlatformWithName(name).value();
    executor_ = platform->ExecutorForDevice(0).value();
  }

  void RunAddI32Kernel(const KernelLoaderSpec& spec) {
    TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
    TF_ASSERT_OK_AND_ASSIGN(auto add, AddI32Kernel::Create(executor_, spec));

    int64_t length = 4;
    int64_t byte_length = sizeof(int32_t) * length;

    // Prepare arguments: a=1, b=2, c=0
    DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(length, 0);
    DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(length, 0);
    DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(length, 0);

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto binary, GetGpuTestKernelsFatbin(executor_->GetPlatform()->Name()));
  KernelLoaderSpec spec =
      KernelLoaderSpec::CreateCudaCubinInMemorySpec(binary, "AddI32", 3);
  RunAddI32Kernel(spec);
}

TEST_F(GpuKernelTest, LoadAndRunKernelFromSymbol) {
  TF_ASSERT_OK_AND_ASSIGN(
      KernelLoaderSpec spec,
      GetAddI32TestKernelSpec(executor_->GetPlatform()->id()));
  RunAddI32Kernel(spec);
}

TEST_F(GpuKernelTest, LoadAndRunKernelFromSymbolWithCustomArgsPacking) {
  constexpr int64_t kArraySize = 4;
  constexpr int64_t kArraySizeBytes = sizeof(int32_t) * kArraySize;

  // Prepare arguments: in=10, out=0
  DeviceAddress<int32_t> in =
      executor_->AllocateArray<int32_t>(kArraySize, /*memory_space=*/0);
  DeviceAddress<int32_t> out =
      executor_->AllocateArray<int32_t>(kArraySize, /*memory_space=*/0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream());
  TF_ASSERT_OK(stream->Memset32(&in, 10, kArraySizeBytes));
  TF_ASSERT_OK(stream->MemZero(&out, kArraySizeBytes));

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                          GetIncrementBy5I32TestKernelSpecWithCustomArgsPacking(
                              executor_->GetPlatform()->id()));
  TF_ASSERT_OK_AND_ASSIGN(auto kernel, executor_->LoadKernel(spec));
  TF_ASSERT_OK(kernel->Launch(
      ThreadDim(), BlockDim(4),
      /*cluster_dims=*/std::nullopt, stream.get(),
      KernelArgsDeviceAddressArray({in, out}, /*shared_memory_bytes=*/0)));

  // Copy data back to host and verify that the output is 5 + 10 = 15.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, kArraySizeBytes));
  EXPECT_THAT(dst, Each(15));
}

TEST_F(GpuKernelTest, ArrayArgByValue) {
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto kernel, LoadCopyTestKernel(executor_));

  constexpr int64_t kLength = 16;

  DeviceAddress<char> dst = executor_->AllocateArray<char>(kLength, 0);
  TF_ASSERT_OK(stream->MemZero(&dst, kLength));

  std::array<std::byte, 16> storage;
  int i = 0;
  for (auto& element : storage) {
    element = static_cast<std::byte>(i++);
  }

  // Launch kernel.
  auto args = stream_executor::PackKernelArgs(/*shmem_bytes=*/0, dst, storage);
  TF_ASSERT_OK(kernel->Launch(ThreadDim(), BlockDim(), stream.get(), *args));

  // Copy data back to host.
  std::byte dst_host[16] = {};
  TF_ASSERT_OK(stream->Memcpy(dst_host, dst, kLength));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_THAT(dst_host, ::testing::ElementsAreArray(storage));
}

TEST_F(GpuKernelTest, TmaLoadAndRunKernelFromPtx) {
  if (!IsTmaAvailableForDevice(executor_->GetDeviceDescription())) {
    GTEST_SKIP() << "TMA is not supported on this platform.";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto tma_kernel,
                          TmaKernel::Create(executor_, GetTmaPtxKernelSpec()));

  auto get_tma_descriptor_from_proto =
      [](absl::string_view proto) -> absl::StatusOr<TmaDescriptor> {
    return TmaDescriptor::FromProto(
        ParseTextProtoOrDie<TmaDescriptorProto>(proto));
  };

  TF_ASSERT_OK_AND_ASSIGN(TmaDescriptor arg0_desc,
                          get_tma_descriptor_from_proto(
                              R"pb(
                                element_size: 2
                                global_dims: 512
                                global_dims: 1024
                                global_strides: 1024
                                box_dims: 64
                                box_dims: 16
                                element_strides: 1
                                element_strides: 1
                                swizzle: SWIZZLE_BYTES128
                                l2_promotion: L2_PROMOTION_BYTES128
                              )pb"));

  TF_ASSERT_OK_AND_ASSIGN(TmaDescriptor arg1_desc,
                          get_tma_descriptor_from_proto(
                              R"pb(
                                element_size: 2
                                global_dims: 1024
                                global_dims: 512
                                global_strides: 2048
                                box_dims: 16
                                box_dims: 128
                                element_strides: 1
                                element_strides: 1
                                swizzle: SWIZZLE_BYTES32
                                l2_promotion: L2_PROMOTION_BYTES128
                              )pb"));

  TF_ASSERT_OK_AND_ASSIGN(TmaDescriptor arg2_desc,
                          get_tma_descriptor_from_proto(
                              R"pb(
                                element_size: 4
                                global_dims: 1024
                                global_dims: 1024
                                global_strides: 4096
                                box_dims: 16
                                box_dims: 16
                                element_strides: 1
                                element_strides: 1
                                swizzle: SWIZZLE_BYTES64
                                l2_promotion: L2_PROMOTION_BYTES128
                              )pb"));

  DeviceAddress<int16_t> mem0 = executor_->AllocateArray<int16_t>(512 * 1024);
  DeviceAddress<int16_t> mem1 = executor_->AllocateArray<int16_t>(1024 * 512);
  DeviceAddress<int32_t> mem2 = executor_->AllocateArray<int32_t>(1024 * 1024);

  TF_ASSERT_OK_AND_ASSIGN(auto tma0,
                          executor_->CreateTensorMap(arg0_desc, mem0.opaque()));
  TF_ASSERT_OK_AND_ASSIGN(auto tma1,
                          executor_->CreateTensorMap(arg1_desc, mem1.opaque()));
  TF_ASSERT_OK_AND_ASSIGN(auto tma2,
                          executor_->CreateTensorMap(arg2_desc, mem2.opaque()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<KernelArgs> packed_args,
      stream_executor::PackKernelArgs(
          absl::Span<const stream_executor::KernelArg>({tma0, tma1, tma2}),
          tma_kernel->metadata()));
  TF_ASSERT_OK(
      tma_kernel->Launch(ThreadDim(), BlockDim(), stream.get(), *packed_args));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
}

}  // namespace
}  // namespace stream_executor::gpu
