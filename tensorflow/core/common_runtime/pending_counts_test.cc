/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include <unordered_map>

#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(PendingCounts, Simple) {
  const int C = 300;
  PendingCounts c(C);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(id, id, id);
  }
  for (int id = 0; id < C; id++) {
    EXPECT_EQ(c.pending(id), id);
    EXPECT_EQ(c.dead_count(id), 0);
  }

  for (int id = 0; id < C; id++) {
    c.increment_dead_count(id);
    // The dead count is no longer updated once pending is 0.
    EXPECT_EQ(c.dead_count(id), (id == 0) ? 0 : 1);
  }

  EXPECT_EQ(c.decrement_pending(1, 1), 0);
  EXPECT_EQ(c.decrement_pending(3, 1), 2);
  EXPECT_EQ(c.decrement_pending(3, 1), 1);
  c.decrement_pending(5, 1);
  c.decrement_pending(5, 3);
  c.decrement_pending(170, 1);
  c.decrement_pending(170, 13);
  EXPECT_EQ(c.pending(1), 0);
  EXPECT_EQ(c.pending(3), 1);
  EXPECT_EQ(c.pending(5), 1);
  EXPECT_EQ(c.pending(170), 156);
}

TEST(PendingCounts, InitializeFrom) {
  const int C = 300;
  PendingCounts c(C);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(id, id, id);
  }
  PendingCounts c2(C);
  c2.InitializeFrom(c);
  for (int id = 0; id < C; id++) {
    EXPECT_EQ(c.pending(id), c2.pending(id));
    EXPECT_EQ(c.dead_count(id), c2.dead_count(id));
  }
}

TEST(PendingCounts, SmallPendingLargeDead) {
  PendingCounts c(1);
  c.set_initial_count(0, 1, 10);
  EXPECT_EQ(c.pending(0), 1);
  EXPECT_EQ(c.dead_count(0), 0);
  for (int i = 1; i <= 10; i++) {
    c.increment_dead_count(0);
    EXPECT_EQ(c.dead_count(0), i);
  }
  EXPECT_EQ(c.pending(0), 1);
  EXPECT_EQ(c.decrement_pending(0, 1), 0);
  EXPECT_EQ(c.dead_count(0), 10);
}

TEST(PendingCounts, MarkLiveShowsUpAsCount) {
  PendingCounts c(2);
  for (int id = 0; id < 2; id++) {
    // Test for both packed and large.
    int count = (id == 0) ? 5 : 15;
    c.set_initial_count(id, count, 4);
    EXPECT_EQ(c.pending(id), count);
    c.mark_live(id);
    EXPECT_EQ(c.pending(id), count - 1);
    // mark_live should be idempotent
    c.mark_live(id);
    EXPECT_EQ(c.pending(id), count - 1);

    c.decrement_pending(id, count - 1);
    EXPECT_EQ(c.pending(id), 0);

    // mark_live should be idempotent
    c.mark_live(id);
    EXPECT_EQ(c.pending(id), 0);
    c.mark_started(id);
    c.mark_live(id);
    EXPECT_EQ(c.pending(id), 0);
    c.mark_completed(id);
    c.mark_live(id);
    EXPECT_EQ(c.pending(id), 0);
  }
}

TEST(PendingCounts, StateIsCorrect) {
  const int C = 20;
  PendingCounts c(C);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(id, id, id);
  }
  for (int id = 0; id < C; id++) {
    while (c.pending(id) > 0) {
      EXPECT_EQ(c.node_state(id), PendingCounts::PENDING_NOTREADY);
      c.decrement_pending(id, 1);
    }
    EXPECT_EQ(c.node_state(id), PendingCounts::PENDING_READY);
    c.mark_started(id);
    EXPECT_EQ(c.node_state(id), PendingCounts::STARTED);
    c.mark_completed(id);
    EXPECT_EQ(c.node_state(id), PendingCounts::COMPLETED);
  }
}

}  // namespace tensorflow
