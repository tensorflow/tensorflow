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

#include "tensorflow/core/common_runtime/cost_measurement_registry.h"

#include "absl/time/time.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestCostMeasurement : public CostMeasurement {
 public:
  absl::Duration GetTotalCost() override { return absl::ZeroDuration(); }
};

REGISTER_COST_MEASUREMENT("test_cost_measurement", TestCostMeasurement);

TEST(CostMeasurementRegistryTest, Basic) {
  std::unique_ptr<const CostMeasurement> test_cost_messurement =
      CostMeasurementRegistry::CreateByNameOrNull(
          "unregistered_cost_measurement");
  EXPECT_EQ(test_cost_messurement, nullptr);

  test_cost_messurement =
      CostMeasurementRegistry::CreateByNameOrNull("test_cost_measurement");
  EXPECT_NE(test_cost_messurement, nullptr);
}

TEST(CostMeasurementRegistryDeathTest, CrashWhenRegisterTwice) {
  const auto creator = []() {
    return absl::make_unique<TestCostMeasurement>();
  };
  EXPECT_DEATH(CostMeasurementRegistry::RegisterCostMeasurement(
                   "test_cost_measurement", creator),
               "CostMeasurement test_cost_measurement is registered twice.");
}

}  // namespace
}  // namespace tensorflow
