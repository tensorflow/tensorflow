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
  PendingCounts::Layout layout;
  std::vector<PendingCounts::Handle> h(C);
  for (int id = 0; id < C; id++) {
    h[id] = layout.CreateHandle(id, id);
  }

  PendingCounts c(layout);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(h[id], id);
  }
  for (int id = 0; id < C; id++) {
    EXPECT_EQ(c.pending(h[id]), id);
    EXPECT_EQ(c.dead_count(h[id]), 0);
  }

  for (int id = 0; id < C; id++) {
    c.increment_dead_count(h[id]);
    // The dead count is no longer updated once pending is 0.
    EXPECT_EQ(c.dead_count(h[id]), (id == 0) ? 0 : 1);
  }

  EXPECT_EQ(c.decrement_pending(h[1], 1), 0);
  EXPECT_EQ(c.decrement_pending(h[3], 1), 2);
  EXPECT_EQ(c.decrement_pending(h[3], 1), 1);
  c.decrement_pending(h[5], 1);
  c.decrement_pending(h[5], 3);
  c.decrement_pending(h[170], 1);
  c.decrement_pending(h[170], 13);
  EXPECT_EQ(c.pending(h[1]), 0);
  EXPECT_EQ(c.pending(h[3]), 1);
  EXPECT_EQ(c.pending(h[5]), 1);
  EXPECT_EQ(c.pending(h[170]), 156);
}

TEST(PendingCounts, CopyConstructor) {
  const int C = 300;
  PendingCounts::Layout layout;
  std::vector<PendingCounts::Handle> h(C);
  for (int id = 0; id < C; id++) {
    h[id] = layout.CreateHandle(id, id);
  }
  PendingCounts c(layout);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(h[id], id);
  }
  PendingCounts c2(c);
  for (int id = 0; id < C; id++) {
    EXPECT_EQ(c.pending(h[id]), c2.pending(h[id]));
    EXPECT_EQ(c.dead_count(h[id]), c2.dead_count(h[id]));
  }
}

TEST(PendingCounts, MarkLiveShowsUpAsCount) {
  PendingCounts::Layout layout;
  PendingCounts::Handle handles[2];
  handles[0] = layout.CreateHandle(5, 4);
  handles[1] = layout.CreateHandle(15, 4);
  for (int id = 0; id < 2; id++) {
    PendingCounts::Handle h = handles[id];
    // Test for both packed and large.
    int count = (id == 0) ? 5 : 15;

    PendingCounts c(layout);
    c.set_initial_count(h, count);
    EXPECT_EQ(c.pending(h), count);
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), count - 1);
    // mark_live should be idempotent
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), count - 1);

    c.decrement_pending(h, count - 1);
    EXPECT_EQ(c.pending(h), 0);

    // mark_live should be idempotent
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), 0);
    c.mark_started(h);
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), 0);
    c.mark_completed(h);
    c.mark_live(h);
    EXPECT_EQ(c.pending(h), 0);
  }
}

TEST(PendingCounts, StateIsCorrect) {
  const int C = 20;
  PendingCounts::Layout layout;
  std::vector<PendingCounts::Handle> handles(C);
  for (int id = 0; id < C; id++) {
    handles[id] = layout.CreateHandle(id, id);
  }
  PendingCounts c(layout);
  for (int id = 0; id < C; id++) {
    c.set_initial_count(handles[id], id);
  }

  for (int id = 0; id < C; id++) {
    PendingCounts::Handle h = handles[id];
    while (c.pending(h) > 0) {
      EXPECT_EQ(c.node_state(h), PendingCounts::PENDING_NOTREADY);
      c.decrement_pending(h, 1);
    }
    EXPECT_EQ(c.node_state(h), PendingCounts::PENDING_READY);
    c.mark_started(h);
    EXPECT_EQ(c.node_state(h), PendingCounts::STARTED);
    c.mark_completed(h);
    EXPECT_EQ(c.node_state(h), PendingCounts::COMPLETED);
  }
}

TEST(PendingCounts, AdjustForActivation) {
  PendingCounts::Layout layout;
  PendingCounts::Handle handles[2];
  handles[0] = layout.CreateHandle(5, 4);
  handles[1] = layout.CreateHandle(15, 4);
  for (int id = 0; id < 2; id++) {
    PendingCounts::Handle h = handles[id];
    // Test for both packed and large.
    int count = (id == 0) ? 5 : 15;
    int pending, dead;

    PendingCounts c(layout);
    c.set_initial_count(h, count);
    EXPECT_EQ(c.pending(h), count);

    // Don't increment the dead count this time
    c.adjust_for_activation(h, false, &pending, &dead);
    EXPECT_EQ(c.pending(h), count - 1);
    EXPECT_EQ(c.pending(h), pending);
    EXPECT_EQ(c.dead_count(h), 0);
    EXPECT_EQ(c.dead_count(h), dead);

    // Increment the dead count this time
    c.adjust_for_activation(h, true, &pending, &dead);
    EXPECT_EQ(c.pending(h), count - 2);
    EXPECT_EQ(c.pending(h), pending);
    EXPECT_EQ(c.dead_count(h), dead);
    EXPECT_EQ(c.dead_count(h), 1);
  }
}

}  // namespace tensorflow
