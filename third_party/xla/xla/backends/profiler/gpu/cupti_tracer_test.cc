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

#include "xla/backends/profiler/gpu/cupti_tracer.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_driver_cbid.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"
#include "xla/backends/profiler/gpu/cuda_test.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/backends/profiler/gpu/mock_cupti.h"
#include "xla/tsl/profiler/utils/time_utils.h"

namespace xla {
namespace profiler {
namespace test {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrictMock;

auto SetTimestampAndReturnSuccess(uint64_t timestamp) {
  return DoAll(SetArgPointee<1>(timestamp), Return(CUPTI_SUCCESS));
}

class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer(CuptiInterface* cupti_interface)
      : CuptiTracer(cupti_interface) {}
};

class CuptiTracerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_GT(CuptiTracer::NumGpus(), 0) << "No devices found";
    auto mock_cupti = std::make_unique<StrictMock<MockCupti>>();
    mock_ = mock_cupti.get();
    cupti_error_manager_ =
        std::make_unique<CuptiErrorManager>(std::move(mock_cupti));
    cupti_tracer_ =
        std::make_unique<TestableCuptiTracer>(cupti_error_manager_.get());
    cupti_wrapper_ = std::make_unique<CuptiWrapper>();

    CuptiTracerCollectorOptions collector_options;
    collector_options.num_gpus = CuptiTracer::NumGpus();
    uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
    uint64_t start_walltime_ns = tsl::profiler::GetCurrentTimeNanos();
    cupti_collector_ = CreateCuptiCollector(
        collector_options, start_walltime_ns, start_gputime_ns);
  }

  void EnableProfiling(const CuptiTracerOptions& options) {
    cupti_tracer_->Enable(options, cupti_collector_.get()).IgnoreError();
  }

  void DisableProfiling() { cupti_tracer_->Disable(); }

  bool CuptiDisabled() const { return cupti_error_manager_->Disabled(); }

  CuptiTracerOptions KernelTraceOptions() {
    CuptiTracerOptions options;
    options.activities_selected = {CUPTI_ACTIVITY_KIND_KERNEL};
    options.cbids_selected = {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel};
    return options;
  }

  void ExpectSuccessfulV1KernelTrace(CUpti_SubscriberHandle subscriber) {
    const int resource_cb_count = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
    EXPECT_CALL(*mock_,
                EnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                EnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, SetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityUsePerThreadBuffer())
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityRegisterCallbacks(_, _))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                EnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                EnableCallback(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, Unsubscribe(subscriber))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
        .WillOnce(Return(CUPTI_SUCCESS));
  }

  void ExpectV2ResourceCallbacks(CUpti_SubscriberHandle subscriber,
                                 uint32_t enable) {
    const int count = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
    if (count == 0) {
      return;
    }
    EXPECT_CALL(*mock_,
                EnableCallback(enable, subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(count)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }

  void ExpectV2KernelCallback(CUpti_SubscriberHandle subscriber,
                              uint32_t enable) {
    EXPECT_CALL(*mock_,
                EnableCallback(enable, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .WillOnce(Return(CUPTI_SUCCESS));
  }

  void ExpectV2KernelSession(CUpti_SubscriberHandle subscriber,
                             uint64_t preflight_timestamp,
                             uint64_t fallback_stop_timestamp,
                             uint64_t stop_timestamp,
                             CUptiResult stop_timestamp_status = CUPTI_SUCCESS,
                             bool expect_preflight_timestamp = true) {
    if (expect_preflight_timestamp) {
      EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
          .WillOnce(SetTimestampAndReturnSuccess(preflight_timestamp));
    }
    ExpectV2ResourceCallbacks(subscriber, /*enable=*/1);
    ExpectV2KernelCallback(subscriber, /*enable=*/1);
    EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(subscriber))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(subscriber, _, _))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_,
                ActivityEnableV2(subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
        .WillOnce(SetTimestampAndReturnSuccess(fallback_stop_timestamp));
    ExpectV2ResourceCallbacks(subscriber, /*enable=*/0);
    ExpectV2KernelCallback(subscriber, /*enable=*/0);
    EXPECT_CALL(*mock_,
                ActivityDisableV2(subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
        .WillOnce(Return(CUPTI_SUCCESS));
    if (stop_timestamp_status == CUPTI_SUCCESS) {
      EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
          .WillOnce(SetTimestampAndReturnSuccess(stop_timestamp));
    } else {
      EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
          .WillOnce(Return(stop_timestamp_status));
    }
  }

  StrictMock<MockCupti>* mock_;
  std::unique_ptr<TestableCuptiTracer> cupti_tracer_;
  std::unique_ptr<CuptiErrorManager> cupti_error_manager_;
  std::unique_ptr<CuptiWrapper> cupti_wrapper_;
  std::unique_ptr<CuptiTraceCollector> cupti_collector_;
};

class CuptiV2SubscribeFallbackTest
    : public CuptiTracerTest,
      public ::testing::WithParamInterface<CUptiResult> {};

class CuptiV2TimestampFallbackTest
    : public CuptiTracerTest,
      public ::testing::WithParamInterface<CUptiResult> {};

TEST_P(CuptiV2SubscribeFallbackTest, FallsBackToV1) {
  EXPECT_FALSE(CuptiDisabled());

  // CuptiWrapper returns CUPTI_ERROR_NOT_SUPPORTED without calling
  // cuptiSubscribe_v2 when either cuptiSubscribe_v2 or cuptiGetTimestamp_v2 is
  // unavailable. CUPTI_ERROR_UNKNOWN from cuptiSubscribe_v2 is also safe to
  // fall back from because no V2 subscriber was successfully created.
  const CUptiResult result = GetParam();
  auto* const v1_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _)).WillOnce(Return(result));
  EXPECT_CALL(*mock_, Subscribe(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(v1_subscriber), Return(CUPTI_SUCCESS)));
  ExpectSuccessfulV1KernelTrace(v1_subscriber);

  CuptiTracerOptions options = KernelTraceOptions();
  EnableProfiling(options);
  DisableProfiling();

  EXPECT_FALSE(CuptiDisabled());
}

INSTANTIATE_TEST_SUITE_P(NonfatalErrors, CuptiV2SubscribeFallbackTest,
                         ::testing::Values(CUPTI_ERROR_NOT_SUPPORTED,
                                           CUPTI_ERROR_UNKNOWN));

TEST_P(CuptiV2TimestampFallbackTest, FallsBackToV1) {
  EXPECT_FALSE(CuptiDisabled());

  const CUptiResult result = GetParam();
  auto* const v2_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  auto* const v1_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{2});
  ::testing::InSequence in_sequence;
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(v2_subscriber), Return(CUPTI_SUCCESS)));
  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .WillOnce(Return(result));
  EXPECT_CALL(*mock_, Unsubscribe(v2_subscriber))
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, Subscribe(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(v1_subscriber), Return(CUPTI_SUCCESS)));
  ExpectSuccessfulV1KernelTrace(v1_subscriber);

  CuptiTracerOptions options = KernelTraceOptions();
  EnableProfiling(options);
  DisableProfiling();

  EXPECT_FALSE(CuptiDisabled());
}

// NOT_SUPPORTED covers an unavailable V2 timestamp capability. UNKNOWN
// preserves fallback for an unclassified failure in the optional V2 path.
// Both are safe to fall back from because the tracer unsubscribes V2 before
// retrying with V1.
INSTANTIATE_TEST_SUITE_P(NonfatalErrors, CuptiV2TimestampFallbackTest,
                         ::testing::Values(CUPTI_ERROR_NOT_SUPPORTED,
                                           CUPTI_ERROR_UNKNOWN));

TEST_F(CuptiTracerTest,
       FatalTimestampV2FailureDoesNotUnsubscribeFreshSubscriberTwice) {
  EXPECT_FALSE(CuptiDisabled());

  auto* const v2_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(v2_subscriber), Return(CUPTI_SUCCESS)));
  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .WillOnce(Return(CUPTI_ERROR_INVALID_PARAMETER));
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_INVALID_PARAMETER, _))
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // The error-manager undo stack owns this newly created subscriber and must
  // be the only code path that unsubscribes it.
  EXPECT_CALL(*mock_, Unsubscribe(v2_subscriber))
      .WillOnce(Return(CUPTI_SUCCESS));

  CuptiTracerOptions options;
  EnableProfiling(options);

  EXPECT_TRUE(CuptiDisabled());
}

TEST_F(CuptiTracerTest, PreparesFreshV2SubscriberBeforeEachSessionTimestamp) {
  EXPECT_FALSE(CuptiDisabled());

  CuptiTracerOptions options = KernelTraceOptions();

  for (uintptr_t session = 1; session <= 2; ++session) {
    ::testing::InSequence in_sequence;
    auto* const subscriber = reinterpret_cast<CUpti_SubscriberHandle>(session);
    EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
        .WillOnce(DoAll(SetArgPointee<0>(subscriber), Return(CUPTI_SUCCESS)));
    EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
        .WillOnce(SetTimestampAndReturnSuccess(session * 10));

    absl::Status prepare_status =
        cupti_tracer_->PrepareForProfilerStart(options);
    ASSERT_TRUE(prepare_status.ok()) << prepare_status;

    EXPECT_CALL(*mock_, GetTimestampV2(subscriber, _))
        .WillOnce(SetTimestampAndReturnSuccess(session * 10 + 1));
    EXPECT_EQ(cupti_tracer_->GetTimestampForSubscriber(), session * 10 + 1);

    ExpectV2KernelSession(subscriber, /*preflight_timestamp=*/0,
                          /*fallback_stop_timestamp=*/session * 10 + 2,
                          /*stop_timestamp=*/session * 10 + 3, CUPTI_SUCCESS,
                          /*expect_preflight_timestamp=*/false);
    EXPECT_CALL(*mock_, Unsubscribe(subscriber))
        .WillOnce(Return(CUPTI_SUCCESS));

    EnableProfiling(options);
    DisableProfiling();
    EXPECT_EQ(cupti_collector_->GetTracingEndTimeNs(), session * 10 + 3);
  }

  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiTracerTest,
       FatalStopTimestampDoesNotUnsubscribeFreshSubscriberTwice) {
  ::testing::InSequence in_sequence;
  auto* const subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .WillOnce(DoAll(SetArgPointee<0>(subscriber), Return(CUPTI_SUCCESS)));
  ExpectV2KernelSession(subscriber, /*preflight_timestamp=*/1,
                        /*fallback_stop_timestamp=*/2,
                        /*stop_timestamp=*/0, CUPTI_ERROR_INVALID_PARAMETER);
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_INVALID_PARAMETER, _))
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  EXPECT_CALL(*mock_,
              ActivityDisableV2(subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .WillOnce(Return(CUPTI_SUCCESS));
  ExpectV2KernelCallback(subscriber, /*enable=*/0);
  ExpectV2ResourceCallbacks(subscriber, /*enable=*/0);
  EXPECT_CALL(*mock_, Unsubscribe(subscriber)).WillOnce(Return(CUPTI_SUCCESS));

  CuptiTracerOptions options = KernelTraceOptions();
  EnableProfiling(options);
  DisableProfiling();
  EXPECT_EQ(cupti_collector_->GetTracingEndTimeNs(), 2);
  EXPECT_TRUE(CuptiDisabled());
  // A stale handle would make this retry issue an unexpected second
  // Unsubscribe call through the disabled error manager.
  EnableProfiling(options);
}

}  // namespace
}  // namespace test
}  // namespace profiler
}  // namespace xla
