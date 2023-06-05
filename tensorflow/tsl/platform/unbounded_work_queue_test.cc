/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/platform/unbounded_work_queue.h"

#include "absl/memory/memory.h"
#include "tensorflow/tsl/platform/random.h"
#include "tensorflow/tsl/platform/blocking_counter.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

class UnboundedWorkQueueTest : public ::testing::Test {
 protected:
  UnboundedWorkQueueTest()
      : work_queue_(
            absl::make_unique<UnboundedWorkQueue>(Env::Default(), "test")) {}
  ~UnboundedWorkQueueTest() override = default;

  void RunMultipleCopiesOfClosure(const int num_closures,
                                  std::function<void()> fn) {
    for (int i = 0; i < num_closures; ++i) {
      work_queue_->Schedule([this, fn]() {
        fn();
        mutex_lock l(mu_);
        ++closure_count_;
        cond_var_.notify_all();
      });
    }
  }

  void BlockUntilClosuresDone(const int num_closures) {
    mutex_lock l(mu_);
    while (closure_count_ < num_closures) {
      cond_var_.wait(l);
    }
  }

  void ResetQueue() { work_queue_.reset(); }

  int NumClosuresExecuted() {
    mutex_lock l(mu_);
    return closure_count_;
  }

 private:
  mutex mu_;
  int closure_count_ TF_GUARDED_BY(mu_) = 0;
  condition_variable cond_var_;
  std::unique_ptr<UnboundedWorkQueue> work_queue_;
};

TEST_F(UnboundedWorkQueueTest, SingleClosure) {
  constexpr int num_closures = 1;
  RunMultipleCopiesOfClosure(num_closures, []() {});
  BlockUntilClosuresDone(num_closures);
}

TEST_F(UnboundedWorkQueueTest, MultipleClosures) {
  constexpr int num_closures = 10;
  RunMultipleCopiesOfClosure(num_closures, []() {});
  BlockUntilClosuresDone(num_closures);
}

TEST_F(UnboundedWorkQueueTest, MultipleClosuresSleepingRandomly) {
  constexpr int num_closures = 1000;
  RunMultipleCopiesOfClosure(num_closures, []() {
    Env::Default()->SleepForMicroseconds(random::New64() % 10);
  });
  BlockUntilClosuresDone(num_closures);
}

TEST_F(UnboundedWorkQueueTest, NestedClosures) {
  constexpr int num_closures = 10;
  // Run `num_closures` closures, each of which runs `num_closures` closures.
  RunMultipleCopiesOfClosure(num_closures, [=]() {
    RunMultipleCopiesOfClosure(num_closures, []() {});
  });
  BlockUntilClosuresDone(num_closures * num_closures + num_closures);
}

TEST_F(UnboundedWorkQueueTest, RacyDestructor) {
  constexpr int num_closures = 100;
  // Run `num_closures` closures, then delete `work_queue_`.
  RunMultipleCopiesOfClosure(num_closures, []() {});
  ResetQueue();
  EXPECT_LE(NumClosuresExecuted(), num_closures);
}

}  // namespace
}  // namespace tsl
