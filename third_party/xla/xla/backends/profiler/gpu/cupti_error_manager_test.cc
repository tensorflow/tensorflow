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

#if GOOGLE_CUDA

#include "xla/backends/profiler/gpu/cupti_error_manager.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "xla/backends/profiler/gpu/cuda_test.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/backends/profiler/gpu/mock_cupti.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/test.h"

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
    cupti_tracer_->Enable(option, cupti_collector_.get());
  }

  void DisableProfiling() { cupti_tracer_->Disable(); }

  bool CuptiDisabled() const { return cupti_error_manager_->Disabled(); }

  void RunGpuApp() {
    MemCopyH2D();
    PrintfKernel(/*iters=*/10);
    Synchronize();
    MemCopyD2H();
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
  EXPECT_CALL(*mock_, Subscribe(_, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Subscribe));
  const int cb_enable_times = IsCudaNewEnoughForGraphTraceTest() ? 4 : 1;
  EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
      .Times(cb_enable_times)
      .InSequence(s1)
      .WillRepeatedly(
          Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableCallback));
  EXPECT_CALL(*mock_, SetThreadIdType(_))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::SetThreadIdType));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBuffer())
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(),
                       &CuptiWrapper::ActivityUsePerThreadBuffer));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacks(_, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(),
                       &CuptiWrapper::ActivityRegisterCallbacks));
  EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL))
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
      .WillRepeatedly(
          Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableCallback));
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Unsubscribe));

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
  EXPECT_CALL(*mock_, Subscribe(_, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Subscribe));
  const int cb_enable_times = IsCudaNewEnoughForGraphTraceTest() ? 3 : 0;
  if (cb_enable_times > 0) {
    EXPECT_CALL(*mock_, EnableCallback(1, _, _, _))
        .Times(cb_enable_times)
        .InSequence(s1)
        .WillRepeatedly(
            Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableCallback));
  }
  EXPECT_CALL(*mock_, EnableDomain(1, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableDomain));
  EXPECT_CALL(*mock_, SetThreadIdType(_))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::SetThreadIdType));
  EXPECT_CALL(*mock_, ActivityUsePerThreadBuffer())
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(),
                       &CuptiWrapper::ActivityUsePerThreadBuffer));
  EXPECT_CALL(*mock_, ActivityRegisterCallbacks(_, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(),
                       &CuptiWrapper::ActivityRegisterCallbacks));
  EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::ActivityEnable));
  EXPECT_CALL(*mock_, ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2))
      .InSequence(s1)
      .WillOnce(Return(CUPTI_ERROR_UNKNOWN));  // injected error
  // CuptiErrorManager::ResultString()
  EXPECT_CALL(*mock_, GetResultString(CUPTI_ERROR_UNKNOWN, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::GetResultString));
  // CuptiErrorManager::UndoAndDisable()
  EXPECT_CALL(*mock_, ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::ActivityDisable));
  EXPECT_CALL(*mock_, EnableDomain(0, _, _))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableDomain));
  if (cb_enable_times > 0) {
    EXPECT_CALL(*mock_, EnableCallback(0, _, _, _))
        .Times(cb_enable_times)
        .InSequence(s1)
        .WillRepeatedly(
            Invoke(cupti_wrapper_.get(), &CuptiWrapper::EnableCallback));
  }
  EXPECT_CALL(*mock_, Unsubscribe(_))
      .InSequence(s1)
      .WillOnce(Invoke(cupti_wrapper_.get(), &CuptiWrapper::Unsubscribe));

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

}  // namespace test
}  // namespace profiler
}  // namespace xla

#endif  // GOOGLE_CUDA
