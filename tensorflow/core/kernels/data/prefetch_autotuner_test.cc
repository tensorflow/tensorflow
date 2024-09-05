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

#include <vector>

#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(PrefetchAutotuner, Disabled) {
  auto ram_manager = std::make_shared<model::RamBudgetManager>(/*budget=*/100);
  PrefetchAutotuner t(2, 0, ram_manager);
  t.SetElementSize(1);
  EXPECT_EQ(2, t.buffer_limit());
  t.RecordConsumption(0);
  t.RecordConsumption(2);
  t.RecordConsumption(0);
  t.RecordConsumption(2);
  EXPECT_EQ(2, t.buffer_limit());
}

TEST(PrefetchAutotuner, Enabled) {
  auto ram_manager = std::make_shared<model::RamBudgetManager>(/*budget=*/100);
  PrefetchAutotuner t(model::kAutotune, 0, ram_manager);
  t.SetElementSize(1);
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
  auto ram_manager = std::make_shared<model::RamBudgetManager>(/*budget=*/100);
  PrefetchAutotuner t(model::kAutotune, 0, ram_manager);
  t.SetElementSize(1);
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
  auto ram_manager = std::make_shared<model::RamBudgetManager>(/*budget=*/100);
  PrefetchAutotuner t(model::kAutotune, 2, ram_manager);
  t.SetElementSize(1);
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

TEST(PrefetchAutotuner, RespectRamManager) {
  auto ram_manager = std::make_shared<model::RamBudgetManager>(/*budget=*/200);
  PrefetchAutotuner t(model::kAutotune, 2, ram_manager);
  t.SetElementSize(50);
  EXPECT_EQ(2, t.buffer_limit());
  // Buffer can grow once since 4*50 <= 200.
  t.RecordConsumption(2);
  t.RecordConsumption(0);
  EXPECT_EQ(4, t.buffer_limit());
  // Buffer is not allowed to grow again since it would exceed memory budget.
  t.RecordConsumption(4);
  t.RecordConsumption(0);
  EXPECT_EQ(4, t.buffer_limit());
}

TEST(PrefetchAutotuner, RespectRamManagerWhenThereIsModelAllocation) {
  int64_t model_allocation = 100000;
  auto ram_manager = std::make_shared<model::RamBudgetManager>(
      /*budget=*/200 + model_allocation);
  // 200 + `model_allocation` - `model_allocation` => 200
  ASSERT_TRUE(ram_manager->RequestModelAllocation(model_allocation));
  PrefetchAutotuner t(model::kAutotune, 2, ram_manager);
  t.SetElementSize(50);
  EXPECT_EQ(2, t.buffer_limit());
  // Buffer can grow once since 4*50 <= 200.
  t.RecordConsumption(2);
  t.RecordConsumption(0);
  EXPECT_EQ(4, t.buffer_limit());
  // Buffer is not allowed to grow again since it would exceed memory budget
  // i.e. 8 * 50 > 200
  t.RecordConsumption(4);
  t.RecordConsumption(0);
  EXPECT_EQ(4, t.buffer_limit());
  // Reset model allocation to zero
  ASSERT_TRUE(ram_manager->RequestModelAllocation(0));
  t.RecordConsumption(4);
  t.RecordConsumption(0);
  // The buffer limit should grow to 8 since model allocation is reset to 0
  EXPECT_EQ(8, t.buffer_limit());
  t.RecordConsumption(8);
  t.RecordConsumption(0);
  // Still plenty of space, the buffer limit should grow
  EXPECT_EQ(16, t.buffer_limit());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
