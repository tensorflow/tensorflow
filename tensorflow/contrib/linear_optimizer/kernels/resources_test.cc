/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/contrib/linear_optimizer/kernels/resources.h"

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Operators for testing convenience (for EQ and NE GUnit macros).
bool operator==(const DataByExample::Data& lhs,
                const DataByExample::Data& rhs) {
  return lhs.dual == rhs.dual &&                //
         lhs.primal_loss == rhs.primal_loss &&  //
         lhs.dual_loss == rhs.dual_loss &&      //
         lhs.example_weight == rhs.example_weight;
}

bool operator!=(const DataByExample::Data& lhs,
                const DataByExample::Data& rhs) {
  return !(lhs == rhs);
}

class DataByExampleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const string solver_uuid = "TheSolver";
    ASSERT_TRUE(resource_manager_
                    .LookupOrCreate<DataByExample>(
                        container_, solver_uuid, &data_by_example_,
                        [&, this](DataByExample** ret) {
                          *ret = new DataByExample(container_, solver_uuid);
                          return Status::OK();
                        })
                    .ok());
  }

  void TearDown() override {
    data_by_example_->Unref();
    ASSERT_TRUE(resource_manager_.Cleanup(container_).ok());
  }

  // Accessors and mutators to private members of DataByExample for better
  // testing.
  static size_t VisitChunkSize() { return DataByExample::kVisitChunkSize; }
  void InsertReservedEntryUnlocked() NO_THREAD_SAFETY_ANALYSIS {
    data_by_example_->data_by_key_[{0, 0}];
  }

  const string container_ = "TheContainer";
  ResourceMgr resource_manager_;
  DataByExample* data_by_example_ = nullptr;
};

TEST_F(DataByExampleTest, MakeKeyIsCollisionResistent) {
  const DataByExample::Key key = DataByExample::MakeKey("TheExampleId");
  EXPECT_NE(key.first, key.second);
  EXPECT_NE(key.first & 0xFFFFFFFF, key.second);
}

TEST_F(DataByExampleTest, ElementAccessAndMutation) {
  const DataByExample::Key key1 = DataByExample::MakeKey("TheExampleId1");
  EXPECT_EQ(DataByExample::Data(), data_by_example_->Get(key1));

  DataByExample::Data data1;
  data1.dual = 1.0f;
  data_by_example_->Set(key1, data1);
  EXPECT_EQ(data1, data_by_example_->Get(key1));

  const DataByExample::Key key2 = DataByExample::MakeKey("TheExampleId2");
  EXPECT_NE(data_by_example_->Get(key1), data_by_example_->Get(key2));
}

TEST_F(DataByExampleTest, VisitEmpty) {
  size_t num_elements = 0;
  ASSERT_TRUE(
      data_by_example_
          ->Visit([&](const DataByExample::Data& data) { ++num_elements; })
          .ok());
  EXPECT_EQ(0, num_elements);
}

TEST_F(DataByExampleTest, VisitMany) {
  const size_t kNumElements = 2 * VisitChunkSize() + 1;
  for (size_t i = 0; i < kNumElements; ++i) {
    DataByExample::Data data;
    data.dual = static_cast<float>(i);
    data_by_example_->Set(DataByExample::MakeKey(strings::StrCat(i)), data);
  }
  size_t num_elements = 0;
  double total_dual = 0;
  ASSERT_TRUE(data_by_example_
                  ->Visit([&](const DataByExample::Data& data) {
                    ++num_elements;
                    total_dual += data.dual;
                  })
                  .ok());
  EXPECT_EQ(kNumElements, num_elements);
  EXPECT_DOUBLE_EQ(
      // 0 + 1 + ... + (N-1) = (N-1)*N/2
      (kNumElements - 1) * kNumElements / 2.0, total_dual);
}

TEST_F(DataByExampleTest, VisitUnavailable) {
  // Populate enough entries so that Visiting will be chunked.
  for (size_t i = 0; i < 2 * VisitChunkSize(); ++i) {
    data_by_example_->Get(DataByExample::MakeKey(strings::StrCat(i)));
  }

  struct Condition {
    mutex mu;
    bool c GUARDED_BY(mu) = false;
    condition_variable cv;
  };
  auto signal = [](Condition* const condition) {
    mutex_lock l(condition->mu);
    condition->c = true;
    condition->cv.notify_all();
  };
  auto wait = [](Condition* const condition) {
    mutex_lock l(condition->mu);
    while (!condition->c) {
      condition->cv.wait(l);
    }
  };

  Condition paused_visit;     // Signaled after a Visit has paused.
  Condition updated_data;     // Signaled after data has been updated.
  Condition completed_visit;  // Signaled after a Visit has completed.

  thread::ThreadPool thread_pool(Env::Default(), "test", 2 /* num_threads */);
  Status status;
  size_t num_visited = 0;
  thread_pool.Schedule([&] {
    status = data_by_example_->Visit([&](const DataByExample::Data& unused) {
      ++num_visited;
      if (num_visited == VisitChunkSize()) {
        // Safe point to mutate the data structure without a lock below.
        signal(&paused_visit);
        wait(&updated_data);
      }
    });
    signal(&completed_visit);
  });
  thread_pool.Schedule([&, this] {
    wait(&paused_visit);
    InsertReservedEntryUnlocked();
    signal(&updated_data);
  });
  wait(&completed_visit);
  EXPECT_TRUE(errors::IsUnavailable(status));
}

}  // namespace tensorflow
