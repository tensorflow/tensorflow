/* Copyright 2015 Google Inc. All Rights Reserved.

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

  for (int id = 0; id < C; id++) {
    c.increment_dead_count(id);
    EXPECT_EQ(c.dead_count(id), 1);
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
  PendingCounts c(3);
  c.set_initial_count(1, 4, 4);
  EXPECT_EQ(c.pending(1), 4);
  c.mark_live(1);
  EXPECT_EQ(c.pending(1), 5);
  // mark_live should be idempotent
  c.mark_live(1);
  EXPECT_EQ(c.pending(1), 5);

  c.decrement_pending(1, 2);
  EXPECT_EQ(c.pending(1), 3);
}

}  // namespace tensorflow
