/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace tsl {
namespace profiler {
namespace {

// Note: These tests should ideally be run separately if they interfere due to
// global state like AnnotationStack::generation_. They are marked as exclusive
// in the BUILD file.
void RunTestWithSequence(const std::vector<int>& order) {
  AnnotationStack::Enable(true);

  // threads vector removed, using ThreadPool instead.
  std::vector<std::vector<int64_t>> collected_ids(6);
  std::atomic<int> start_count{0};
  std::atomic<int> finished_count{0};
  std::atomic<int> new_generation_start{0};

  auto push_work = [&](int idx) {
    while (start_count.load() != idx) {
    }
    AnnotationStack::PushAnnotation("level1");
    // Enable next thread to start its work if applicable
    start_count.fetch_add(1);

    AnnotationStack::PushAnnotation("level2-1");
    AnnotationStack::PopAnnotation();
    AnnotationStack::PushAnnotation("level2-2");
    auto ids = AnnotationStack::GetScopeRangeIds();
    collected_ids[idx] = std::vector<int64_t>(ids.begin(), ids.end());
    AnnotationStack::PopAnnotation();
    AnnotationStack::PopAnnotation();

    finished_count.fetch_add(1);
    while (finished_count.load() < 6) {
    }

    // waiting for new generation to start
    while (new_generation_start.load() == 0) {
    }
    // in fact, new version
    AnnotationStack::PushAnnotation("v2-level-1");
    AnnotationStack::PushAnnotation("v2-level-2");
    ids = AnnotationStack::GetScopeRangeIds();
    collected_ids[idx].insert(collected_ids[idx].end(), ids.begin(), ids.end());
    AnnotationStack::PopAnnotation();
    AnnotationStack::PopAnnotation();
  };

  auto dummy_work = [&](int idx) {
    while (start_count.load() != idx) {
    }
    // Enable next thread to start its work if applicable
    start_count.fetch_add(1);

    // Do some dummy work
    volatile int a = 0;
    for (int i = 0; i < 1000; ++i) {
      a += i;
    }
    (void)a;

    finished_count.fetch_add(1);
    while (finished_count.load() < 6) {
    }
  };

  {
    thread::ThreadPool pool(tsl::Env::Default(), "testpool", 6);
    // Schedule work in the specified order
    for (int i = 0; i < 6; ++i) {
      int type = order[i];
      if (type == 0) {  // Push work
        pool.Schedule([i, &push_work]() { push_work(i); });
      } else {  // Dummy work
        pool.Schedule([i, &dummy_work]() { dummy_work(i); });
      }
    }

    while (finished_count.load() < 6) {
    }
    AnnotationStack::Enable(false);

    AnnotationStack::Enable(true);
    new_generation_start.store(1);
  }  // Destructor of pool runs here and blocks until all tasks fully finish!

  AnnotationStack::Enable(false);

  // Verify results for push threads
  for (uint64_t i = 0, tid = 1; i < 6; ++i) {
    if (order[i] == 0) {
      EXPECT_GT(collected_ids[i].size(), 1);
      uint64_t id1 = collected_ids[i][0];
      uint64_t tid1 = (id1 >> 48);
      // Check expected thread id.
      EXPECT_EQ(tid, tid1);

      for (int j = 1; j < collected_ids[i].size(); ++j) {
        uint64_t id2 = collected_ids[i][j];
        uint64_t tid2 = (id2 >> 48);
        // Tids in the same cpu thread should be the same across different
        // pushes. They should be the same across different generations as well.
        EXPECT_EQ(tid1, tid2)
            << "thread idx: " << i << " scope range idx in results: j: " << j;

        // Range ID (bits 0-47) should be different or incremented.
        // as TID part is same, we can directly compare the full ID.
        EXPECT_NE(id1, id2)
            << "thread idx: " << i << " scope range idx in results: j: " << j;
      }
      tid++;
    }
  }

  std::vector<uint16_t> tids;
  for (int i = 0; i < 6; ++i) {
    if (order[i] == 0 && !collected_ids[i].empty()) {
      uint64_t id = static_cast<uint64_t>(collected_ids[i][0]);
      tids.push_back((id >> 48) & 0x7FFF);
    }
  }
}

TEST(AnnotationStackTest, MultiThreadSequence1) {
  // Order: 3 push threads followed by 3 dummy threads
  RunTestWithSequence({0, 0, 0, 1, 1, 1});
}

TEST(AnnotationStackTest, MultiThreadSequence2) {
  // Order: 3 dummy threads followed by 3 push threads
  RunTestWithSequence({1, 1, 1, 0, 0, 0});
}

TEST(AnnotationStackTest, MultiThreadSequence3) {
  // Interleaved
  RunTestWithSequence({0, 1, 0, 1, 0, 1});
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
