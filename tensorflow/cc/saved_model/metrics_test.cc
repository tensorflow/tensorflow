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

#include "tensorflow/cc/saved_model/metrics.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace metrics {
// The value of the cells for each metric persists across tests.

TEST(MetricsTest, TestSavedModelWrite) {
  EXPECT_EQ(SavedModelWriteApi("foo").value(), 0);
  SavedModelWriteApi("foo").IncrementBy(1);
  EXPECT_EQ(SavedModelWriteApi("foo").value(), 1);

  EXPECT_EQ(SavedModelWrite("1").value(), 0);
  SavedModelWrite("1").IncrementBy(1);
  EXPECT_EQ(SavedModelWrite("1").value(), 1);
}

TEST(MetricsTest, TestSavedModelRead) {
  SavedModelReadApi("bar").IncrementBy(1);
  EXPECT_EQ(SavedModelReadApi("bar").value(), 1);
  SavedModelRead("2").IncrementBy(1);
  EXPECT_EQ(SavedModelRead("2").value(), 1);

  SavedModelReadApi("baz").IncrementBy(1);
  EXPECT_EQ(SavedModelReadApi("baz").value(), 1);
  SavedModelRead("2").IncrementBy(1);
  EXPECT_EQ(SavedModelRead("2").value(), 2);
}

TEST(MetricsTest, TestCheckpointRead) {
  EXPECT_EQ(CheckpointReadDuration("foo").value().num(), 0);
  CheckpointReadDuration("foo").Add(100);
  EXPECT_EQ(CheckpointReadDuration("foo").value().num(), 1);
}

TEST(MetricsTest, TestCheckpointWrite) {
  EXPECT_EQ(CheckpointWriteDuration("foo").value().num(), 0);
  CheckpointWriteDuration("foo").Add(100);
  EXPECT_EQ(CheckpointWriteDuration("foo").value().num(), 1);
}

TEST(MetricsTest, TestTrainingTimeSaved) {
  EXPECT_EQ(TrainingTimeSaved("foo").value(), 0);
  TrainingTimeSaved("foo").IncrementBy(100);
  EXPECT_EQ(TrainingTimeSaved("foo").value(), 100);
}

}  // namespace metrics
}  // namespace tensorflow
