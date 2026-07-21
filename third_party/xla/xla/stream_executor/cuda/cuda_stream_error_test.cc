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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {
namespace {

class CudaStreamTest : public ::testing::Test {
 public:
  CudaExecutor* executor_;

 private:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                            stream_executor::PlatformManager::PlatformWithId(
                                stream_executor::cuda::kCudaPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                            platform->ExecutorForDevice(0));
    executor_ = reinterpret_cast<CudaExecutor*>(executor);
  }
};

TEST_F(CudaStreamTest, StreamErrorPropagatesToCallback) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaStream> stream,
      CudaStream::Create(executor_, /*priority=*/std::nullopt));
  static constexpr absl::string_view kPtx = R"(
      .version 4.2
      .target sm_50
      .address_size 64
      .visible .entry IllegalAccess() {
        .reg .u64 %addr;
        mov.u64 %addr, 0x0;
        st.u64 [%addr], %addr;
        ret;
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> kernel,
      xla::gpu::CreateKernel("IllegalAccess", 0, kPtx, stream->parent(), 0));
  std::pair<tsl::Promise<void>, tsl::Future<void>> promise_and_future =
      tsl::MakePromise<>();
  auto& promise = promise_and_future.first;
  auto& future = promise_and_future.second;
  EXPECT_THAT(xla::gpu::ExecuteKernelOnStream(*kernel, {},
                                              xla::gpu::LaunchDimensions(1, 1),
                                              std::nullopt, stream.get()),
              absl_testing::IsOk());
  EXPECT_THAT(stream->DoHostCallbackWithStatus(
                  [&promise]() {
                    promise.Set();
                    return absl::OkStatus();
                  },
                  [&promise](absl::Status s) { promise.Set(s); }),
              absl_testing::IsOk());
  absl::Status result = future.Await();
  EXPECT_THAT(result, absl_testing::StatusIs(
                          absl::StatusCode::kInternal,
                          testing::HasSubstr("CUDA_ERROR_ILLEGAL_ADDRESS")));
}

TEST_F(CudaStreamTest, StreamSuccessPropagatesToCallback) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaStream> stream,
      CudaStream::Create(executor_, /*priority=*/std::nullopt));
  static constexpr absl::string_view kPtx = R"(
      .version 4.2
      .target sm_50
      .address_size 64
      .visible .entry NoOp() {
        ret;
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> kernel,
      xla::gpu::CreateKernel("NoOp", 0, kPtx, stream->parent(), 0));
  std::pair<tsl::Promise<void>, tsl::Future<void>> promise_and_future =
      tsl::MakePromise<>();
  auto& promise = promise_and_future.first;
  auto& future = promise_and_future.second;
  EXPECT_THAT(xla::gpu::ExecuteKernelOnStream(*kernel, {},
                                              xla::gpu::LaunchDimensions(1, 1),
                                              std::nullopt, stream.get()),
              absl_testing::IsOk());
  EXPECT_THAT(stream->DoHostCallbackWithStatus(
                  [&promise]() {
                    promise.Set();
                    return absl::OkStatus();
                  },
                  [&promise](absl::Status s) { promise.Set(s); }),
              absl_testing::IsOk());
  absl::Status result = future.Await();
  EXPECT_THAT(result, absl_testing::IsOk());
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
