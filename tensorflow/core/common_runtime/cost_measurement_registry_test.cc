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

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kTestCostName[] = "test";

class TestCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::ZeroDuration(); }
  absl::string_view GetCostType() const override { return kTestCostName; }
};

REGISTER_COST_MEASUREMENT(kTestCostName, TestCostMeasurement);

TEST(CostMeasurementRegistryTest, Basic) {
  const CostMeasurement::Context context;
  std::unique_ptr<const CostMeasurement> test_cost_measurement =
      CostMeasurementRegistry::CreateByNameOrNull("unregistered", context);
  EXPECT_EQ(test_cost_measurement, nullptr);

  test_cost_measurement =
      CostMeasurementRegistry::CreateByNameOrNull(kTestCostName, context);
  EXPECT_NE(test_cost_measurement, nullptr);
}

TEST(CostMeasurementRegistryDeathTest, CrashWhenRegisterTwice) {
  const auto creator = [](const CostMeasurement::Context& context) {
    return absl::make_unique<TestCostMeasurement>(context);
  };
  EXPECT_DEATH(
      CostMeasurementRegistry::RegisterCostMeasurement(kTestCostName, creator),
      absl::StrCat("CostMeasurement ", kTestCostName, " is registered twice."));
}

}  // namespace
}  // namespace tensorflow
