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

#include "tensorflow/core/kernels/data/prefetch_autotuner.h"

#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(PrefetchAutotuner, Disabled) {
  PrefetchAutotuner t(2, 0);
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(0);
  t.RecordConsumption(2);
  t.RecordConsumption(0);
  t.RecordConsumption(2);
  EXPECT_EQ(2, t.buffer_limit());
}

TEST(PrefetchAutotuner, Enabled) {
  PrefetchAutotuner t(model::kAutotune, 0);
  EXPECT_EQ(1, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to stay the same.
  EXPECT_EQ(1, t.buffer_limit());
  t.RecordConsumption(1);
  EXPECT_EQ(1, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(2);
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(1);
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, t.buffer_limit());
  t.RecordConsumption(4);
  EXPECT_EQ(4, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(8, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(8, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(8, t.buffer_limit());
}

TEST(PrefetchAutotuner, EnabledSteady) {
  PrefetchAutotuner t(model::kAutotune, 0);
  EXPECT_EQ(1, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(1, t.buffer_limit());
  t.RecordConsumption(1);
  EXPECT_EQ(1, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(2);
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, t.buffer_limit());

  // Never reach zero again.
  std::vector<size_t> consumption_values = {2, 3, 1, 4, 1, 2, 3, 1};
  for (int i = 0; i < consumption_values.size(); ++i) {
    t.RecordConsumption(consumption_values[i]);
    EXPECT_EQ(4, t.buffer_limit())
        << "Failed at index " << i << " with value: " << consumption_values[i];
  }
}

TEST(PrefetchAutotuner, StartWithMin) {
  PrefetchAutotuner t(model::kAutotune, 2);
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(2);  // Expect buffer limit to stay the same!
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, t.buffer_limit());
  t.RecordConsumption(4);  // Expect buffer limit to stay the same!
  EXPECT_EQ(4, t.buffer_limit());
  t.RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(8, t.buffer_limit());

  // Never reach zero again.
  std::vector<size_t> consumption_values = {3, 5, 7, 1, 4, 6, 8, 3, 5, 1, 2, 4};
  for (int i = 0; i < consumption_values.size(); ++i) {
    t.RecordConsumption(consumption_values[i]);
    EXPECT_EQ(8, t.buffer_limit())
        << "Failed at index " << i << " with value: " << consumption_values[i];
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
