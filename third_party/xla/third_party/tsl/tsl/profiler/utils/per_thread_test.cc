/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/utils/per_thread.h"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

/*
The timeline used in the test are as below:

t0 ------------------------  t1 ------------------- t2 -------------------- t3

t0/t3 begin/end
t1 : profiling start
t2 : profiling stop

Profiling_0 start will be called at t0, profiling_0 stop right before t1.
after profile_0 stopped, profiling_1 will start right before t1, profiling_1
will stopped right after t2.

On the timeline, various threads will be started/stopped, some of the thread
will use the per-thread data, some of them never use the per-thread data.
A thread id will be identified as combination of the following:
    the start_stage :
    first_use_stage : the first time when the thread use the per-thread data
    exit_stage:
All combination will be covered in the test to verify the results.
And id used for thread are represented by the three decimal digits for
the above stages enum value.

ThreadSyncControl and thread join will be used to make sure profiling
start/stop happens the right time, and each thread behave as designed.
*/

enum ProfilingStage {
  kBeforeT1 = 1,
  kDuringT1T2 = 2,
  kAfterT2 = 3,
  kNever = 4
};

struct ThreadSyncControl {
  ThreadSyncControl()
      : could_start_profiling_1(4),
        could_stop_profiling_1(6),
        could_exit_all(6) {}

  absl::Notification profiling_1_started;
  absl::Notification profiling_1_stopped;
  absl::Notification exiting_all;

  absl::BlockingCounter could_start_profiling_1;
  absl::BlockingCounter could_stop_profiling_1;
  absl::BlockingCounter could_exit_all;
};

static ThreadSyncControl& GetSyncContols() {
  static ThreadSyncControl* control = new ThreadSyncControl();
  return *control;
}

void ThreadMain(int32_t id, ProfilingStage firstUseStage,
                ProfilingStage exitStage) {
  if (firstUseStage == kBeforeT1) {
    auto& td = PerThread<int32_t>::Get();
    td = id;
    GetSyncContols().could_start_profiling_1.DecrementCount();
  }
  if (exitStage == kBeforeT1) {
    return;
  }
  GetSyncContols().profiling_1_started.WaitForNotification();

  if (firstUseStage == kDuringT1T2) {
    auto& td = PerThread<int32_t>::Get();
    td = id;
    GetSyncContols().could_stop_profiling_1.DecrementCount();
  }
  if (exitStage == kDuringT1T2) {
    return;
  }
  GetSyncContols().profiling_1_stopped.WaitForNotification();

  if (firstUseStage == kAfterT2) {
    auto& td = PerThread<int32_t>::Get();
    td = id;
    GetSyncContols().could_exit_all.DecrementCount();
  }
  if (exitStage == kAfterT2) {
    return;
  }
  GetSyncContols().exiting_all.WaitForNotification();
}

class ThreadFactory {
 public:
  ThreadFactory() : threads_existing_at_(kNever + 1) {}

  void Start(int32_t id, ProfilingStage firstUseStage,
             ProfilingStage exitStage) {
    std::string name = absl::StrCat("thread_", id);
    threads_existing_at_[exitStage].emplace_back(absl::WrapUnique(
        Env::Default()->StartThread(ThreadOptions(), name, [=]() {
          ThreadMain(id, firstUseStage, exitStage);
        })));
  }

  void StopAllAt(ProfilingStage exitStage) {
    threads_existing_at_[exitStage].clear();
  }

 private:
  std::vector<std::list<std::unique_ptr<tsl::Thread>>> threads_existing_at_;
};

using ::testing::ElementsAre;
using ::testing::WhenSorted;

TEST(PerThreadRecordingTest, Lifecycles) {
  auto get_ids = [](std::vector<std::shared_ptr<int32_t>>& threads_data) {
    std::vector<int> threads_values;
    for (const auto& ptd : threads_data) {
      threads_values.push_back(*ptd);
    }
    return threads_values;
  };

  ThreadFactory thread_factory;

  auto threads_data = PerThread<int32_t>::StartRecording();
  auto threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, ::testing::SizeIs(0));

  thread_factory.Start(111, kBeforeT1, kBeforeT1);
  thread_factory.Start(112, kBeforeT1, kDuringT1T2);
  thread_factory.Start(113, kBeforeT1, kAfterT2);
  thread_factory.Start(114, kBeforeT1, kNever);

  thread_factory.Start(122, kDuringT1T2, kDuringT1T2);
  thread_factory.Start(123, kDuringT1T2, kAfterT2);
  thread_factory.Start(124, kDuringT1T2, kNever);

  thread_factory.Start(133, kAfterT2, kAfterT2);
  thread_factory.Start(134, kAfterT2, kNever);

  // These thread will never initialize the Per Thread data
  thread_factory.Start(141, kNever, kBeforeT1);
  thread_factory.Start(142, kNever, kDuringT1T2);
  thread_factory.Start(143, kNever, kAfterT2);
  thread_factory.Start(144, kNever, kNever);

  GetSyncContols().could_start_profiling_1.Wait();
  thread_factory.StopAllAt(kBeforeT1);

  threads_data = PerThread<int32_t>::StopRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, WhenSorted(ElementsAre(111, 112, 113, 114)));

  // Start again, thread 111 already exit
  threads_data = PerThread<int32_t>::StartRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, WhenSorted(ElementsAre(112, 113, 114)));

  GetSyncContols().profiling_1_started.Notify();

  thread_factory.Start(222, kDuringT1T2, kDuringT1T2);
  thread_factory.Start(223, kDuringT1T2, kAfterT2);
  thread_factory.Start(224, kDuringT1T2, kNever);

  thread_factory.Start(233, kAfterT2, kAfterT2);
  thread_factory.Start(234, kAfterT2, kNever);

  thread_factory.Start(242, kNever, kDuringT1T2);
  thread_factory.Start(243, kNever, kAfterT2);
  thread_factory.Start(244, kNever, kNever);

  GetSyncContols().could_stop_profiling_1.Wait();
  thread_factory.StopAllAt(kDuringT1T2);

  threads_data = PerThread<int32_t>::StopRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values, WhenSorted(ElementsAre(112, 113, 114, 122, 123,
                                                     124, 222, 223, 224)));

  threads_data = PerThread<int32_t>::StartRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values,
              WhenSorted(ElementsAre(113, 114, 123, 124, 223, 224)));

  GetSyncContols().profiling_1_stopped.Notify();

  thread_factory.Start(333, kAfterT2, kAfterT2);
  thread_factory.Start(334, kAfterT2, kNever);

  thread_factory.Start(343, kNever, kAfterT2);
  thread_factory.Start(344, kNever, kNever);

  GetSyncContols().could_exit_all.Wait();
  thread_factory.StopAllAt(kAfterT2);

  threads_data = PerThread<int32_t>::StopRecording();
  threads_values = get_ids(threads_data);
  EXPECT_THAT(threads_values,
              WhenSorted(ElementsAre(113, 114, 123, 124, 133, 134, 223, 224,
                                     233, 234, 333, 334)));

  GetSyncContols().exiting_all.Notify();
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
