/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/profiler_factory.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_lock.h"
#include "tsl/profiler/lib/profiler_passes.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

class TestProfiler : public ProfilerInterface {
 public:
  absl::Status Start() override { return absl::OkStatus(); }
  absl::Status Stop() override { return absl::OkStatus(); }
  absl::Status CollectData(tensorflow::profiler::XSpace*) override {
    return absl::OkStatus();
  }
};

std::unique_ptr<ProfilerInterface> TestFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return std::make_unique<TestProfiler>();
}

TEST(ProfilerFactoryTest, FactoryFunctionPointer) {
  ClearRegisteredProfilersForTest();
  RegisterProfilerFactory(&TestFactoryFunction);
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

TEST(ProfilerFactoryTest, FactoryLambda) {
  ClearRegisteredProfilersForTest();
  RegisterProfilerFactory([](const tensorflow::ProfileOptions& options) {
    return std::make_unique<TestProfiler>();
  });
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

std::unique_ptr<ProfilerInterface> NullFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return nullptr;
}

TEST(ProfilerFactoryTest, FactoryReturnsNull) {
  ClearRegisteredProfilersForTest();
  RegisterProfilerFactory(&NullFactoryFunction);
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_TRUE(profilers.empty());
}

class FactoryClass {
 public:
  explicit FactoryClass(void* ptr) : ptr_(ptr) {}
  FactoryClass(const FactoryClass&) = default;  // copyable
  FactoryClass(FactoryClass&&) = default;       // movable

  std::unique_ptr<ProfilerInterface> CreateProfiler(
      const tensorflow::ProfileOptions& options) const {
    return std::make_unique<TestProfiler>();
  }

 private:
  void* ptr_ TF_ATTRIBUTE_UNUSED = nullptr;
};

TEST(ProfilerFactoryTest, FactoryClassCapturedByLambda) {
  ClearRegisteredProfilersForTest();
  static int token = 42;
  FactoryClass factory(&token);
  RegisterProfilerFactory([factory = std::move(factory)](
                              const tensorflow::ProfileOptions& options) {
    return factory.CreateProfiler(options);
  });
  auto profilers = CreateProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

class TestMultiPassProfiler : public MultiPassProfilerInterface {
 public:
  bool NeedMorePasses() override { return false; }
  absl::Status StartPass() override { return absl::OkStatus(); }
  absl::Status PushRange(absl::string_view name) override {
    return absl::OkStatus();
  }
  absl::Status PopRange() override { return absl::OkStatus(); }
  absl::Status StopPass() override { return absl::OkStatus(); }

  absl::Status Start() override { return absl::OkStatus(); }
  absl::Status Stop() override { return absl::OkStatus(); }
  absl::Status CollectData(tensorflow::profiler::XSpace*) override {
    return absl::OkStatus();
  }
};

std::unique_ptr<MultiPassProfilerInterface> TestMultiPassFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return std::make_unique<TestMultiPassProfiler>();
}

TEST(ProfilerFactoryTest, MultiPassFactoryFunctionPointer) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(&TestMultiPassFactoryFunction);
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

TEST(ProfilerFactoryTest, MultiPassFactoryLambda) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(
      [](const tensorflow::ProfileOptions& options) {
        return std::make_unique<TestMultiPassProfiler>();
      });
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  EXPECT_EQ(profilers.size(), 1);
}

std::unique_ptr<MultiPassProfilerInterface> NullMultiPassFactoryFunction(
    const tensorflow::ProfileOptions& options) {
  return nullptr;
}

TEST(ProfilerFactoryTest, MultiPassFactoryReturnsNull) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(&NullMultiPassFactoryFunction);
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  EXPECT_TRUE(profilers.empty());
}

class TrackingMultiPassProfiler : public MultiPassProfilerInterface {
 public:
  struct Tracker {
    int start_called = 0;
    int stop_called = 0;
    int start_pass_called = 0;
    int stop_pass_called = 0;
    int push_range_called = 0;
    int pop_range_called = 0;
    int collect_data_called = 0;
    bool fail_push_range = false;
    bool fail_pop_range = false;
  };

  explicit TrackingMultiPassProfiler(Tracker* tracker = nullptr)
      : tracker_(tracker) {}

  bool NeedMorePasses() override { return false; }
  absl::Status StartPass() override {
    start_pass_called_++;
    if (tracker_) tracker_->start_pass_called++;
    return absl::OkStatus();
  }
  absl::Status PushRange(absl::string_view name) override {
    push_range_called_++;
    if (tracker_) tracker_->push_range_called++;
    if (fail_push_range_ || (tracker_ && tracker_->fail_push_range)) {
      return absl::InternalError("PushRange failed");
    }
    return absl::OkStatus();
  }
  absl::Status PopRange() override {
    pop_range_called_++;
    if (tracker_) tracker_->pop_range_called++;
    if (fail_pop_range_ || (tracker_ && tracker_->fail_pop_range)) {
      return absl::InternalError("PopRange failed");
    }
    return absl::OkStatus();
  }
  absl::Status StopPass() override {
    stop_pass_called_++;
    if (tracker_) tracker_->stop_pass_called++;
    return absl::OkStatus();
  }

  absl::Status Start() override {
    start_called_++;
    if (tracker_) tracker_->start_called++;
    return absl::OkStatus();
  }
  absl::Status Stop() override {
    stop_called_++;
    if (tracker_) tracker_->stop_called++;
    return absl::OkStatus();
  }
  absl::Status CollectData(tensorflow::profiler::XSpace*) override {
    collect_data_called_++;
    if (tracker_) tracker_->collect_data_called++;
    return absl::OkStatus();
  }

  Tracker* tracker_ = nullptr;
  int start_called_ = 0;
  int stop_called_ = 0;
  int start_pass_called_ = 0;
  int stop_pass_called_ = 0;
  int push_range_called_ = 0;
  int pop_range_called_ = 0;
  int collect_data_called_ = 0;
  bool fail_push_range_ = false;
  bool fail_pop_range_ = false;
};

TEST(ProfilerFactoryTest, MultiPassControllerStopWhilePassActiveStopsPass) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  ASSERT_EQ(profilers.size(), 1);
  ASSERT_TRUE(profilers[0]->Start().ok());
  ASSERT_TRUE(profilers[0]->StartPass().ok());
  EXPECT_EQ(raw_profiler->start_pass_called_, 1);
  EXPECT_EQ(raw_profiler->stop_pass_called_, 0);

  // Calling Stop() while pass is started should automatically stop the pass.
  EXPECT_TRUE(profilers[0]->Stop().ok());
  EXPECT_EQ(raw_profiler->stop_pass_called_, 1);
  EXPECT_EQ(raw_profiler->stop_called_, 1);
}

TEST(ProfilerFactoryTest,
     MultiPassControllerErrorInPassTransitionsStateAndStopsPass) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
  ASSERT_EQ(profilers.size(), 1);
  ASSERT_TRUE(profilers[0]->Start().ok());
  ASSERT_TRUE(profilers[0]->StartPass().ok());
  raw_profiler->fail_push_range_ = true;
  EXPECT_FALSE(profilers[0]->PushRange("fail").ok());

  // StopPass() should still invoke StopPass() on underlying profiler and
  // transition state.
  EXPECT_FALSE(profilers[0]->StopPass().ok());
  EXPECT_EQ(raw_profiler->stop_pass_called_, 1);

  // Stop() should now handle stopping the profiler session cleanly.
  EXPECT_FALSE(profilers[0]->Stop().ok());  // Latched error is returned
  // No call forwarded since error latched.
  EXPECT_EQ(raw_profiler->stop_called_, 0);
}

TEST(ProfilerFactoryTest,
     MultiPassControllerDestructorCleansUpActivePassOnError) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  {
    auto profilers = CreateMultiPassProfilers(tensorflow::ProfileOptions());
    ASSERT_EQ(profilers.size(), 1);
    ASSERT_TRUE(profilers[0]->Start().ok());
    ASSERT_TRUE(profilers[0]->StartPass().ok());
    raw_profiler->fail_push_range_ = true;
    EXPECT_FALSE(profilers[0]->PushRange("fail").ok());
    // Destructor runs when exiting this scope while state_ == kPassStarted with
    // error.
  }
  EXPECT_EQ(raw_profiler->stop_pass_called_, 1);
  EXPECT_EQ(raw_profiler->stop_called_, 1);
}

#if !defined(IS_MOBILE_PLATFORM)
TEST(ProfilerPassesTest, CollectDataReleasesLockOnError) {
  ClearRegisteredMultiPassProfilersForTest();
  RegisterMultiPassProfilerFactory(
      [](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        profiler->fail_push_range_ = true;
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  // Fail PushRange so status_ latches an error.
  ASSERT_FALSE(passes->PushRange("fail").ok());
  ASSERT_FALSE(passes->StopPass().ok());

  // CollectData should fail, but must release the profiler lock!
  tensorflow::profiler::XSpace space;
  EXPECT_FALSE(passes->CollectData(&space).ok());

  // While passes is still in scope, another session must be able to acquire the
  // lock.
  EXPECT_FALSE(ProfilerLock::HasActiveSession());
  auto passes2 = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  EXPECT_TRUE(passes2->Status().ok());
}

TEST(ProfilerPassesTest, PopRangeWithoutActiveRangeFails) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 0);

  // Calling PopRange when no range has been pushed should fail.
  absl::Status status = passes->PopRange();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  // Underlying profiler PopRange should NOT have been called.
  EXPECT_EQ(raw_profiler->pop_range_called_, 0);
}

TEST(ProfilerPassesTest, StopPassPopsAllActiveRanges) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("range1").ok());
  ASSERT_TRUE(passes->PushRange("range2").ok());
  ASSERT_TRUE(passes->PushRange("range3").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 3);
  EXPECT_EQ(raw_profiler->pop_range_called_, 0);

  // Calling StopPass should automatically pop all 3 active ranges.
  EXPECT_TRUE(passes->StopPass().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 3);
}

TEST(ProfilerPassesTest, PushAndPopRangesTrackCount) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("range1").ok());
  ASSERT_TRUE(passes->PushRange("range2").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 2);
  EXPECT_EQ(raw_profiler->pop_range_called_, 0);

  // Pop one range explicitly.
  ASSERT_TRUE(passes->PopRange().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 1);

  // Push another range.
  ASSERT_TRUE(passes->PushRange("range3").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 3);

  // StopPass should pop the 2 remaining active ranges.
  EXPECT_TRUE(passes->StopPass().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 3);
}

TEST(ProfilerPassesTest, PopRangeFailsWhenMorePopsThanPushes) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("range1").ok());
  ASSERT_TRUE(passes->PushRange("range2").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 2);

  // Pop both active ranges explicitly.
  ASSERT_TRUE(passes->PopRange().ok());
  ASSERT_TRUE(passes->PopRange().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 2);

  // Third PopRange should fail since active_range_count_ is 0.
  absl::Status status = passes->PopRange();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  // Underlying pop_range_called_ should not have increased.
  EXPECT_EQ(raw_profiler->pop_range_called_, 2);
}

TEST(ProfilerPassesTest, DestructorPopsRemainingActiveRanges) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler::Tracker tracker;
  RegisterMultiPassProfilerFactory(
      [&tracker](const tensorflow::ProfileOptions& options) {
        return std::make_unique<TrackingMultiPassProfiler>(&tracker);
      });
  {
    auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
    ASSERT_TRUE(passes->StartPass().ok());
    ASSERT_TRUE(passes->PushRange("range1").ok());
    ASSERT_TRUE(passes->PushRange("range2").ok());
    EXPECT_EQ(tracker.push_range_called, 2);
    EXPECT_EQ(tracker.pop_range_called, 0);
  }
  // Destructor should have popped the 2 active ranges.
  EXPECT_EQ(tracker.pop_range_called, 2);
}

TEST(ProfilerPassesTest, CollectDataPopsRemainingActiveRanges) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler::Tracker tracker;
  RegisterMultiPassProfilerFactory(
      [&tracker](const tensorflow::ProfileOptions& options) {
        return std::make_unique<TrackingMultiPassProfiler>(&tracker);
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("range1").ok());
  EXPECT_EQ(tracker.push_range_called, 1);
  EXPECT_EQ(tracker.pop_range_called, 0);
  EXPECT_EQ(tracker.stop_pass_called, 0);
  tensorflow::profiler::XSpace space;
  EXPECT_TRUE(passes->CollectData(&space).ok());
  EXPECT_EQ(tracker.pop_range_called, 1);
  EXPECT_EQ(tracker.stop_pass_called, 1);
  EXPECT_EQ(tracker.collect_data_called, 1);
}

TEST(ProfilerPassesTest, CollectDataAfterStopPassDoesNotCallStopPassAgain) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler::Tracker tracker;
  RegisterMultiPassProfilerFactory(
      [&tracker](const tensorflow::ProfileOptions& options) {
        return std::make_unique<TrackingMultiPassProfiler>(&tracker);
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("range1").ok());
  ASSERT_TRUE(passes->StopPass().ok());
  EXPECT_EQ(tracker.pop_range_called, 1);
  EXPECT_EQ(tracker.stop_pass_called, 1);

  tensorflow::profiler::XSpace space;
  EXPECT_TRUE(passes->CollectData(&space).ok());
  EXPECT_EQ(tracker.pop_range_called, 1);
  EXPECT_EQ(tracker.stop_pass_called, 1);
  EXPECT_EQ(tracker.collect_data_called, 1);
}

TEST(ProfilerPassesTest, MultiPassTracksAndPopsRangesPerPass) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());

  // Pass 1: Push 2 ranges, explicitly pop 1, StopPass should pop the remaining
  // 1.
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("p1_r1").ok());
  ASSERT_TRUE(passes->PushRange("p1_r2").ok());
  ASSERT_TRUE(passes->PopRange().ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 2);
  EXPECT_EQ(raw_profiler->pop_range_called_, 1);
  EXPECT_TRUE(passes->StopPass().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 2);

  // Pass 2: Push 3 ranges, StopPass should pop all 3.
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("p2_r1").ok());
  ASSERT_TRUE(passes->PushRange("p2_r2").ok());
  ASSERT_TRUE(passes->PushRange("p2_r3").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 5);
  EXPECT_EQ(raw_profiler->pop_range_called_, 2);
  EXPECT_TRUE(passes->StopPass().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 5);
}

TEST(ProfilerPassesTest, PushRangeFailureDoesNotIncrementActiveRangeCount) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());

  // Make PushRange fail.
  raw_profiler->fail_push_range_ = true;
  EXPECT_FALSE(passes->PushRange("fail").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 1);

  // StopPass should not call PopRange since active_range_count_ was not
  // incremented.
  EXPECT_FALSE(passes->StopPass().ok());
  EXPECT_EQ(raw_profiler->pop_range_called_, 0);
}

TEST(ProfilerPassesTest, StopPassPropagatesPopAllRangesError) {
  ClearRegisteredMultiPassProfilersForTest();
  TrackingMultiPassProfiler* raw_profiler = nullptr;
  RegisterMultiPassProfilerFactory(
      [&raw_profiler](const tensorflow::ProfileOptions& options) {
        auto profiler = std::make_unique<TrackingMultiPassProfiler>();
        raw_profiler = profiler.get();
        return profiler;
      });
  auto passes = ProfilerPasses::Create(ProfilerPasses::DefaultOptions());
  ASSERT_TRUE(passes->StartPass().ok());
  ASSERT_TRUE(passes->PushRange("range1").ok());
  ASSERT_TRUE(passes->PushRange("range2").ok());
  EXPECT_EQ(raw_profiler->push_range_called_, 2);

  // Make PopRange fail during StopPass -> PopAllRanges.
  raw_profiler->fail_pop_range_ = true;
  absl::Status status = passes->StopPass();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  // Both active ranges should still have been popped and StopPass called.
  EXPECT_EQ(raw_profiler->pop_range_called_, 2);
  EXPECT_EQ(raw_profiler->stop_pass_called_, 1);
}
#endif

}  // namespace
}  // namespace profiler
}  // namespace tsl
