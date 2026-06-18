/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_error_manager.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_driver_cbid.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"
#include "xla/backends/profiler/gpu/cuda_test.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/backends/profiler/gpu/mock_cupti.h"
#include "xla/tsl/profiler/utils/time_utils.h"

namespace xla {
namespace profiler {
namespace test {

using xla::profiler::CuptiInterface;
using xla::profiler::CuptiTracer;
using xla::profiler::CuptiTracerCollectorOptions;
using xla::profiler::CuptiTracerOptions;
using xla::profiler::CuptiWrapper;

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::Sequence;
using ::testing::StrictMock;

bool TimestampV2FallsBackToLegacy(CUptiResult status) {
  return status == CUPTI_ERROR_NOT_SUPPORTED || status == CUPTI_ERROR_UNKNOWN;
}

// Needed to create different cupti tracer for each test cases.
class TestableCuptiTracer : public CuptiTracer {
 public:
  explicit TestableCuptiTracer(CuptiInterface* cupti_interface)
      : CuptiTracer(cupti_interface) {}
};

// CuptiErrorManagerTest verifies that an application is not killed due to an
// unexpected error in the underlying GPU hardware during tracing.
// MockCupti is used to simulate a CUPTI call failure.
class CuptiErrorManagerTest : public ::testing::Test {
 protected:
  CuptiErrorManagerTest() {}

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

  void EnableProfiling(const CuptiTracerOptions& option) {
    cupti_tracer_->Enable(option, cupti_collector_.get()).IgnoreError();
  }

  void DisableProfiling() { cupti_tracer_->Disable(); }

  bool CuptiDisabled() const { return cupti_error_manager_->Disabled(); }

  void RunGpuApp() {
    MemCopyH2D();
    PrintfKernel(/*iters=*/10);
    Synchronize();
    MemCopyD2H();
  }

  void VerifyV2SubscribeRetainedWhenTimestampV2Returns(
      CUptiResult timestamp_status) {
    EXPECT_FALSE(CuptiDisabled());

    Sequence s1;
    auto* const v2_subscriber =
        reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});

    EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
        .InSequence(s1)
        .WillOnce([&](CUpti_SubscriberHandle* subscriber,
                      CUpti_CallbackFunc /*callback*/, void* /*userdata*/) {
          *subscriber = v2_subscriber;
          return CUPTI_SUCCESS;
        });
    EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
        .InSequence(s1)
        .WillOnce(Return(timestamp_status));

    const int resource_cb_count = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
    if (resource_cb_count > 0) {
      EXPECT_CALL(*mock_,
                  EnableCallback(1, v2_subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
          .Times(resource_cb_count)
          .InSequence(s1)
          .WillRepeatedly(Return(CUPTI_SUCCESS));
    }
    EXPECT_CALL(*mock_,
                EnableCallback(1, v2_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .InSequence(s1)
        .WillOnce(Return(CUPTI_SUCCESS));
    if (resource_cb_count > 0) {
      EXPECT_CALL(*mock_,
                  EnableCallback(0, v2_subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
          .Times(resource_cb_count)
          .InSequence(s1)
          .WillRepeatedly(Return(CUPTI_SUCCESS));
    }
    EXPECT_CALL(*mock_,
                EnableCallback(0, v2_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
        .InSequence(s1)
        .WillOnce(Return(CUPTI_SUCCESS));
    EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
        .InSequence(s1)
        .WillOnce(Return(CUPTI_SUCCESS));
    if (TimestampV2FallsBackToLegacy(timestamp_status)) {
      EXPECT_CALL(*mock_, GetTimestamp(_))
          .InSequence(s1)
          .WillOnce([](uint64_t* timestamp) {
            *timestamp = 2;
            return CUPTI_SUCCESS;
          });
    }
    EXPECT_CALL(*mock_, Unsubscribe(v2_subscriber))
        .InSequence(s1)
        .WillOnce(Return(CUPTI_SUCCESS));

    CuptiTracerOptions options;
    options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    EnableProfiling(options);

    EXPECT_FALSE(CuptiDisabled());
    DisableProfiling();
    EXPECT_FALSE(CuptiDisabled());
    if (TimestampV2FallsBackToLegacy(timestamp_status)) {
      EXPECT_EQ(cupti_collector_->GetTracingEndTimeNs(), 2);
    } else {
      EXPECT_EQ(cupti_collector_->GetTracingEndTimeNs(), 0);
    }
  }

  // Pointer to MockCupti passed to CuptiBase constructor.
  // Used to inject failures to be handled by CuptiErrorManager.
  // Wrapped in StrictMock so unexpected calls cause a test failure.
  StrictMock<MockCupti>* mock_;

  // CuptiTracer instance that uses MockCupti instead of CuptiWrapper.
  std::unique_ptr<TestableCuptiTracer> cupti_tracer_ = nullptr;

  std::unique_ptr<CuptiInterface> cupti_error_manager_;

  // CuptiWrapper instance to which mock_ calls are delegated.
  std::unique_ptr<CuptiWrapper> cupti_wrapper_;

  std::unique_ptr<xla::profiler::CuptiTraceCollector> cupti_collector_;
};

// Verifies that failed EnableProfiling() does not kill an application.
TEST_F(CuptiErrorManagerTest, GpuTraceActivityEnableTest) {
  // Enforces the order of execution below.
  Sequence s1;
  // CuptiBase::EnableProfiling()
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle* subscriber,
                   CUpti_CallbackFunc /*callback*/, void* /*userdata*/) {
        *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
        return CUPTI_SUCCESS;
      });
  EXPECT_CALL(*mock_, GetTimestampV2(_, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle /*subscriber*/, uint64_t* timestamp) {
        *timestamp = 1;
        return CUPTI_SUCCESS;
      });
  const int cb_enable_times = IsCudaNewEnoughForGraphTraceTest() ? 6 : 1;
  EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
      .Times(cb_enable_times)
      .InSequence(s1)
      .WillRepeatedly(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(_, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityEnableV2(_, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, EnableCallback(0, _, _, _))
      .Times(cb_enable_times)
      .InSequence(s1)
      .WillRepeatedly(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));

  EXPECT_FALSE(CuptiDisabled());
  CuptiTracerOptions options;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);  // CUPTI call fails due to injected error
  EXPECT_TRUE(CuptiDisabled());

  RunGpuApp();  // Application code runs normally

  EXPECT_TRUE(CuptiDisabled());
  DisableProfiling();  // CUPTI calls are ignored
  EXPECT_TRUE(CuptiDisabled());
}

// Verifies that failed EnableProfiling() does not kill an application.
TEST_F(CuptiErrorManagerTest, GpuTraceAutoEnableTest) {
  EXPECT_FALSE(CuptiDisabled());
  // Enforces the order of execution below.
  Sequence s1;
  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle* subscriber,
                   CUpti_CallbackFunc /*callback*/, void* /*userdata*/) {
        *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
        return CUPTI_SUCCESS;
      });
  EXPECT_CALL(*mock_, GetTimestampV2(_, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle /*subscriber*/, uint64_t* timestamp) {
        *timestamp = 1;
        return CUPTI_SUCCESS;
      });
  const int cb_enable_times = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
  if (cb_enable_times > 0) {
    EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
        .Times(cb_enable_times)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_, EnableDomain(1, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(_, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityEnableV2(_, CUPTI_ACTIVITY_KIND_MEMCPY, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityEnableV2(_, CUPTI_ACTIVITY_KIND_MEMCPY2, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, ActivityDisableV2(_, CUPTI_ACTIVITY_KIND_MEMCPY, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, EnableDomain(0, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  if (cb_enable_times > 0) {
    EXPECT_CALL(*mock_, EnableCallback(0, _, _, _))
        .Times(cb_enable_times)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));

  EXPECT_FALSE(CuptiDisabled());
  CuptiTracerOptions options;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  // options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);  // CUPTI call fails due to injected error
  EXPECT_TRUE(CuptiDisabled());

  RunGpuApp();  // Application code runs normally

  EXPECT_TRUE(CuptiDisabled());
  DisableProfiling();  // CUPTI calls are ignored
  EXPECT_TRUE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest, KeepsV2SubscribeWhenTimestampV2Unsupported) {
  VerifyV2SubscribeRetainedWhenTimestampV2Returns(CUPTI_ERROR_NOT_SUPPORTED);
}

TEST_F(CuptiErrorManagerTest, KeepsV2SubscribeWhenTimestampV2Unknown) {
  VerifyV2SubscribeRetainedWhenTimestampV2Returns(CUPTI_ERROR_UNKNOWN);
}

TEST_F(CuptiErrorManagerTest, KeepsV2SubscribeWhenTimestampV2NotCompatible) {
  VerifyV2SubscribeRetainedWhenTimestampV2Returns(CUPTI_ERROR_NOT_COMPATIBLE);
}

TEST_F(CuptiErrorManagerTest,
       ReusesV2SubscriberAndActivityCallbacksWhenRequested) {
  EXPECT_FALSE(CuptiDisabled());

  Sequence s1;
  auto* const v2_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});

  EXPECT_CALL(*mock_, SubscribeV2(_, _, _))
      .InSequence(s1)
      .WillOnce([&](CUpti_SubscriberHandle* subscriber,
                    CUpti_CallbackFunc /*callback*/, void* /*userdata*/) {
        *subscriber = v2_subscriber;
        return CUPTI_SUCCESS;
      });
  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle /*subscriber*/, uint64_t* timestamp) {
        *timestamp = 1;
        return CUPTI_SUCCESS;
      });

  const int resource_cb_count = IsCudaNewEnoughForGraphTraceTest() ? 5 : 0;
  if (resource_cb_count > 0) {
    EXPECT_CALL(*mock_,
                EnableCallback(1, v2_subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_,
              EnableCallback(1, v2_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                             CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(v2_subscriber))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacksV2(v2_subscriber, _, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_,
              ActivityEnableV2(v2_subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  if (resource_cb_count > 0) {
    EXPECT_CALL(*mock_,
                EnableCallback(0, v2_subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_,
              EnableCallback(0, v2_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                             CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_,
              ActivityDisableV2(v2_subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle /*subscriber*/, uint64_t* timestamp) {
        *timestamp = 2;
        return CUPTI_SUCCESS;
      });

  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle /*subscriber*/, uint64_t* timestamp) {
        *timestamp = 3;
        return CUPTI_SUCCESS;
      });
  if (resource_cb_count > 0) {
    EXPECT_CALL(*mock_,
                EnableCallback(1, v2_subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_,
              EnableCallback(1, v2_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                             CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(v2_subscriber))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_,
              ActivityEnableV2(v2_subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  if (resource_cb_count > 0) {
    EXPECT_CALL(*mock_,
                EnableCallback(0, v2_subscriber, CUPTI_CB_DOMAIN_RESOURCE, _))
        .Times(resource_cb_count)
        .InSequence(s1)
        .WillRepeatedly(Return(CUPTI_SUCCESS));
  }
  EXPECT_CALL(*mock_,
              EnableCallback(0, v2_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                             CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_,
              ActivityDisableV2(v2_subscriber, CUPTI_ACTIVITY_KIND_KERNEL, _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_SUCCESS));
  EXPECT_CALL(*mock_, GetTimestampV2(v2_subscriber, _))
      .InSequence(s1)
      .WillOnce([](CUpti_SubscriberHandle /*subscriber*/, uint64_t* timestamp) {
        *timestamp = 4;
        return CUPTI_SUCCESS;
      });

  CuptiTracerOptions options;
  options.reuse_cupti_v2_subscriber = true;
  options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
  options.cbids_selected.push_back(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  EnableProfiling(options);
  DisableProfiling();
  EnableProfiling(options);
  DisableProfiling();

  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest,
       ActivityUseSystemThreadIdV2NotCompatibleDoesNotDisableCupti) {
  auto* const v2_subscriber =
      reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
  EXPECT_CALL(*mock_, ActivityUseSystemThreadIdV2(v2_subscriber))
      .WillOnce(Return(CUPTI_ERROR_NOT_COMPATIBLE));

  EXPECT_EQ(cupti_error_manager_->ActivityUseSystemThreadIdV2(v2_subscriber),
            CUPTI_ERROR_NOT_COMPATIBLE);
  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest,
       ActivityUsePerThreadBufferV2NotCompatibleDoesNotDisableCupti) {
  EXPECT_CALL(*mock_, ActivityUsePerThreadBufferV2())
      .WillOnce(Return(CUPTI_ERROR_NOT_COMPATIBLE));

  EXPECT_EQ(cupti_error_manager_->ActivityUsePerThreadBufferV2(),
            CUPTI_ERROR_NOT_COMPATIBLE);
  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest, ActivityGetNextRecordV2NotSupportedFallsBack) {
  EXPECT_FALSE(CuptiDisabled());

  Sequence s1;
  uint8_t buffer[1] = {};
  CUpti_Activity activity = {};
  CUpti_Activity* record = nullptr;

  EXPECT_CALL(*mock_, ActivityGetNextRecordV2(_, buffer, sizeof(buffer), _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_NOT_SUPPORTED));
  EXPECT_CALL(*mock_, ActivityGetNextRecord(buffer, sizeof(buffer), _))
      .InSequence(s1)
      .WillOnce([&](uint8_t* /*buffer*/, size_t /*valid_buffer_size_bytes*/,
                    CUpti_Activity** record) {
        *record = &activity;
        return CUPTI_SUCCESS;
      });

  EXPECT_EQ(cupti_error_manager_->ActivityGetNextRecordV2(
                /*subscriber=*/nullptr, buffer, sizeof(buffer), &record),
            CUPTI_SUCCESS);
  EXPECT_EQ(record, &activity);
  EXPECT_FALSE(CuptiDisabled());
}

TEST_F(CuptiErrorManagerTest, ActivityGetNextRecordV2UnknownFallsBack) {
  EXPECT_FALSE(CuptiDisabled());

  Sequence s1;
  uint8_t buffer[1] = {};
  CUpti_Activity activity = {};
  CUpti_Activity* record = nullptr;

  EXPECT_CALL(*mock_, ActivityGetNextRecordV2(_, buffer, sizeof(buffer), _))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));
  EXPECT_CALL(*mock_, ActivityGetNextRecord(buffer, sizeof(buffer), _))
      .InSequence(s1)
      .WillOnce([&](uint8_t* /*buffer*/, size_t /*valid_buffer_size_bytes*/,
                    CUpti_Activity** record) {
        *record = &activity;
        return CUPTI_SUCCESS;
      });

  EXPECT_EQ(cupti_error_manager_->ActivityGetNextRecordV2(
                /*subscriber=*/nullptr, buffer, sizeof(buffer), &record),
            CUPTI_SUCCESS);
  EXPECT_EQ(record, &activity);
  EXPECT_FALSE(CuptiDisabled());
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
