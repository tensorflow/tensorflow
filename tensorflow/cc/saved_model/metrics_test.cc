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

TEST(MetricsTest, TestWrite) {
  EXPECT_EQ(SavedModelWriteApi("foo", "1").value(), 0);
  SavedModelWriteApi("foo", "1").IncrementBy(1);
  EXPECT_EQ(SavedModelWriteApi("foo", "1").value(), 1);

  EXPECT_EQ(SavedModelWrite().value(), 0);
  SavedModelWrite().IncrementBy(1);
  EXPECT_EQ(SavedModelWrite().value(), 1);
}

TEST(MetricsTest, TestRead) {
  SavedModelReadApi("bar", "2").IncrementBy(1);
  EXPECT_EQ(SavedModelReadApi("bar", "2").value(), 1);
  SavedModelRead().IncrementBy(1);
  EXPECT_EQ(SavedModelRead().value(), 1);

  SavedModelReadApi("baz", "2").IncrementBy(1);
  EXPECT_EQ(SavedModelReadApi("baz", "2").value(), 1);
  SavedModelRead().IncrementBy(1);
  EXPECT_EQ(SavedModelRead().value(), 2);
}
}  // namespace metrics
}  // namespace tensorflow
