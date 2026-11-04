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

#include "xla/service/gpu/stream_executor_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/autotuning.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_utils.h"

namespace xla::gpu {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::stream_executor::dnn::ConvolutionKind;

struct Result {
  int64_t run_time_ns;
  int64_t scratch_bytes;

  bool operator==(const Result& other) const {
    return other.run_time_ns == run_time_ns &&
           other.scratch_bytes == scratch_bytes;
  };

  explicit operator AutotuneResult() const {
    AutotuneResult result;
    *result.mutable_run_time() =
        tsl::proto_utils::ToDurationProto(absl::Nanoseconds(run_time_ns));
    result.set_scratch_bytes(scratch_bytes);
    return result;
  }
};

static Result ATRToResult(AutotuneResult atr) {
  return Result{.run_time_ns = absl::ToInt64Nanoseconds(
                    tsl::proto_utils::FromDurationProto(atr.run_time())),
                .scratch_bytes = atr.scratch_bytes()};
}

std::vector<AutotuneResult> Results(const std::vector<Result>& stats) {
  std::vector<AutotuneResult> results;
  for (const auto& s : stats) results.push_back(AutotuneResult(s));
  return results;
}

TEST(StreamExecutorTest, PickBestResult) {
  absl::StatusOr<AutotuneResult> atr;

  atr = PickBestResult(Results({{9000, 0}, {1000, 0}, {16000, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({1000, 0}));

  atr = PickBestResult(Results({{4700, 0}, {4600, 0}, {4500, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({4500, 0}));

  atr = PickBestResult(Results({{4700, 0}, {4600, 2}, {4500, 1}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({4700, 0}));

  atr = PickBestResult(Results({{5000, 1}, {6000, 0}, {7500, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({6000, 0}));
}

TEST(StreamExecutorUtilTest, CudnnConvKindToProto) {
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kBackwardFilter),
            ConvolutionKind::BACKWARD_FILTER);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kBackwardInput),
            ConvolutionKind::BACKWARD_DATA);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForward),
            ConvolutionKind::FORWARD);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForwardActivation),
            ConvolutionKind::FORWARD_BIAS_ACTIVATION);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForwardGraph),
            ConvolutionKind::FORWARD_GRAPH);
}

TEST(StreamExecutorUtilTest, CudnnConvKindFromProto) {
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::BACKWARD_FILTER),
              IsOkAndHolds(CudnnConvKind::kBackwardFilter));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::BACKWARD_DATA),
              IsOkAndHolds(CudnnConvKind::kBackwardInput));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::FORWARD),
              IsOkAndHolds(CudnnConvKind::kForward));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::FORWARD_BIAS_ACTIVATION),
              IsOkAndHolds(CudnnConvKind::kForwardActivation));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::FORWARD_GRAPH),
              IsOkAndHolds(CudnnConvKind::kForwardGraph));

  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::INVALID),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(StreamExecutorUtilTest, GetGpuMutexConcurrentPerDevice) {
  constexpr int kNumDevices = 8;
  constexpr int kThreadsPerDevice = 4;

  std::vector<std::unique_ptr<se::MockStreamExecutor>> executors;
  for (int d = 0; d < kNumDevices; ++d) {
    auto exec = std::make_unique<se::MockStreamExecutor>();
    EXPECT_CALL(*exec, device_ordinal()).WillRepeatedly(::testing::Return(d));
    executors.push_back(std::move(exec));
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "gpu_mutex_test",
                               kNumDevices * kThreadsPerDevice);
  absl::BlockingCounter ready(kNumDevices * kThreadsPerDevice);
  absl::Notification go;
  absl::BlockingCounter done(kNumDevices * kThreadsPerDevice);

  std::vector<absl::Mutex*> mutex_ptrs(kNumDevices * kThreadsPerDevice,
                                       nullptr);

  for (int d = 0; d < kNumDevices; ++d) {
    for (int t = 0; t < kThreadsPerDevice; ++t) {
      int idx = d * kThreadsPerDevice + t;
      pool.Schedule([&, d, idx]() {
        ready.DecrementCount();
        go.WaitForNotification();
        absl::Mutex& mu = GetGpuMutex(executors[d].get());
        {
          absl::MutexLock lock(mu);
          mutex_ptrs[idx] = &mu;
        }
        done.DecrementCount();
      });
    }
  }

  ready.Wait();
  go.Notify();
  done.Wait();

  for (int d = 0; d < kNumDevices; ++d) {
    absl::Mutex* base = mutex_ptrs[d * kThreadsPerDevice];
    ASSERT_NE(base, nullptr);
    for (int t = 1; t < kThreadsPerDevice; ++t) {
      EXPECT_EQ(mutex_ptrs[d * kThreadsPerDevice + t], base);
    }
    for (int other_d = d + 1; other_d < kNumDevices; ++other_d) {
      EXPECT_NE(base, mutex_ptrs[other_d * kThreadsPerDevice]);
    }
  }

  se::MockStreamExecutor neg_exec;
  EXPECT_CALL(neg_exec, device_ordinal()).WillRepeatedly(::testing::Return(-1));
  absl::Mutex& fallback_mu = GetGpuMutex(&neg_exec);
  EXPECT_EQ(&fallback_mu, &GetGpuMutex(&neg_exec));
}

}  // namespace

}  // namespace xla::gpu
