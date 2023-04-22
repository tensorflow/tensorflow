/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/recent_request_ids.h"

#include <algorithm>

#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

Status TrackUnique(int64_t request_id, RecentRequestIds* recent_request_ids) {
  RecvTensorRequest request;
  request.set_request_id(request_id);
  return recent_request_ids->TrackUnique(request_id, "recent_request_ids_test",
                                         request);
}

// request_id 0 is always valid.
TEST(RecentRequestIds, Zero) {
  RecentRequestIds recent_request_ids(1);
  EXPECT_TRUE(TrackUnique(0, &recent_request_ids).ok());
  EXPECT_TRUE(TrackUnique(0, &recent_request_ids).ok());
  EXPECT_TRUE(TrackUnique(0, &recent_request_ids).ok());
}

TEST(RecentRequestIds, Unordered) {
  // Capacity for 6 numbers.
  RecentRequestIds recent_request_ids(6);

  // Some unordered numbers to insert into request_id_set.
  std::vector<int64> numbers = {53754,  23351,  164101, 7476,
                                162432, 130761, 164102};

  // Insert numbers[0..6) and check that all previously inserted numbers remain
  // in the set.
  for (int i = 0; i < 6; ++i) {
    TF_EXPECT_OK(TrackUnique(numbers[i], &recent_request_ids));

    for (int j = 0; j <= i; ++j) {
      EXPECT_FALSE(TrackUnique(numbers[j], &recent_request_ids).ok())
          << "i=" << i << " j=" << j;
    }
  }

  // Insert numbers[6]. Inserting this 7th number should evict the first number
  // from the set. The set should only contain numbers[1..7).
  TF_EXPECT_OK(TrackUnique(numbers[6], &recent_request_ids));
  for (int i = 1; i < 7; ++i) {
    EXPECT_FALSE(TrackUnique(numbers[i], &recent_request_ids).ok())
        << "i=" << i;
  }

  // Insert numbers[0] again. This should succeed because we just evicted it
  // from the set.
  TF_EXPECT_OK(TrackUnique(numbers[0], &recent_request_ids));
}

// Check that the oldest request_id is evicted.
void TestOrdered(int num_request_ids) {
  RecentRequestIds recent_request_ids(num_request_ids);

  // Insert [1..101). The current number and the (num_request_ids - 1) preceding
  // numbers should still be in the set.
  for (int i = 1; i < 101; ++i) {
    TF_EXPECT_OK(TrackUnique(i, &recent_request_ids));

    for (int j = std::max(1, i - num_request_ids + 1); j <= i; ++j) {
      EXPECT_FALSE(TrackUnique(j, &recent_request_ids).ok())
          << "i=" << i << " j=" << j;
    }
  }
}

// Test eviction with various numbers of buckets.
TEST(RecentRequestIds, Ordered2) { TestOrdered(2); }
TEST(RecentRequestIds, Ordered3) { TestOrdered(3); }
TEST(RecentRequestIds, Ordered4) { TestOrdered(4); }
TEST(RecentRequestIds, Ordered5) { TestOrdered(5); }

static void BM_TrackUnique(::testing::benchmark::State& state) {
  RecentRequestIds recent_request_ids(100000);
  RecvTensorRequest request;
  for (auto s : state) {
    TF_CHECK_OK(recent_request_ids.TrackUnique(GetUniqueRequestId(),
                                               "BM_TrackUnique", request));
  }
}

BENCHMARK(BM_TrackUnique);

}  // namespace tensorflow
