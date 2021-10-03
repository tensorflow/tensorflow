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

#include "tensorflow/core/common_runtime/cost_util.h"

#include "tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "tensorflow/core/common_runtime/request_cost_accessor_registry.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestCostMeasurement : public CostMeasurement {
 public:
  absl::Duration GetTotalCost() override { return absl::ZeroDuration(); }
  absl::string_view GetCostType() const override { return "test"; }
};
REGISTER_COST_MEASUREMENT("test", TestCostMeasurement);

class TestRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override { return nullptr; }
};
REGISTER_REQUEST_COST_ACCESSOR("test", TestRequestCostAccessor);

TEST(CreateCostMeasurementTest, Basic) {
  setenv("TF_COST_MEASUREMENT_TYPE", "test", /*overwrite=*/1);
  std::unique_ptr<CostMeasurement> test_cost_measurement =
      CreateCostMeasurement();

  ASSERT_NE(test_cost_measurement, nullptr);
  EXPECT_EQ(test_cost_measurement->GetTotalCost(), absl::ZeroDuration());
  EXPECT_EQ(test_cost_measurement->GetCostType(), "test");
}

TEST(CreateRequestCostAccessorTest, Basic) {
  setenv("TF_REQUEST_COST_ACCESSOR_TYPE", "test", /*overwrite=*/1);
  std::unique_ptr<RequestCostAccessor> test_req_cost_accessor =
      CreateRequestCostAccessor();

  ASSERT_NE(test_req_cost_accessor, nullptr);
  EXPECT_EQ(test_req_cost_accessor->GetRequestCost(), nullptr);
}

}  // namespace
}  // namespace tensorflow
