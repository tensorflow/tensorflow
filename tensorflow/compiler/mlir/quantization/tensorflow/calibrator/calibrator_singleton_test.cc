/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace calibrator {
namespace {

TEST(CalibratorSingletonTest, SimpleMinMax) {
  CalibratorSingleton::ReportMinMax("1", 1.0f, 2.0f);
  absl::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax("1");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(1.0f, min_max.value().first);
  EXPECT_THAT(2.0f, min_max.value().second);

  CalibratorSingleton::ReportMinMax("1", -1.0f, 3.0f);
  min_max = CalibratorSingleton::GetMinMax("1");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(-1.0f, min_max.value().first);
  EXPECT_THAT(3.0f, min_max.value().second);

  CalibratorSingleton::ReportMinMax("1", 3.0f, 5.0f);

  min_max = CalibratorSingleton::GetMinMax("1");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(-1.0f, min_max.value().first);
  EXPECT_THAT(5.0f, min_max.value().second);
}

TEST(CalibratorSingletonTest, DifferentSessions) {
  CalibratorSingleton::ReportMinMax("2", 1.0f, 2.0f);
  absl::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax("2");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(1.0f, min_max.value().first);
  EXPECT_THAT(2.0f, min_max.value().second);

  CalibratorSingleton::ReportMinMax("3", -1.0f, 3.0f);
  min_max = CalibratorSingleton::GetMinMax("3");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(-1.0f, min_max.value().first);
  EXPECT_THAT(3.0f, min_max.value().second);

  CalibratorSingleton::ReportMinMax("2", 3.0f, 5.0f);
  min_max = CalibratorSingleton::GetMinMax("2");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(1.0f, min_max.value().first);
  EXPECT_THAT(5.0f, min_max.value().second);

  min_max = CalibratorSingleton::GetMinMax("3");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(-1.0f, min_max.value().first);
  EXPECT_THAT(3.0f, min_max.value().second);
}

TEST(CalibratorSingletonTest, ClearAndGetEmptyResult) {
  CalibratorSingleton::ReportMinMax("4", 1.0f, 2.0f);
  absl::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax("4");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(1.0f, min_max.value().first);
  EXPECT_THAT(2.0f, min_max.value().second);

  CalibratorSingleton::ClearCollectedInformation();

  min_max = CalibratorSingleton::GetMinMax("4");
  EXPECT_FALSE(min_max.has_value());
}

TEST(CalibratorSingletonTest, ClearDataAndGetResults) {
  CalibratorSingleton::ReportMinMax("5", 1.0f, 2.0f);
  absl::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax("5");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(min_max.value().first, 1.0f);
  EXPECT_THAT(min_max.value().second, 2.0f);

  CalibratorSingleton::ReportMinMax("6", 3.0f, 4.0f);
  min_max = CalibratorSingleton::GetMinMax("6");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(min_max.value().first, 3.0f);
  EXPECT_THAT(min_max.value().second, 4.0f);

  CalibratorSingleton::ClearData("5");

  min_max = CalibratorSingleton::GetMinMax("5");
  EXPECT_FALSE(min_max.has_value());

  min_max = CalibratorSingleton::GetMinMax("6");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_THAT(min_max.value().first, 3.0f);
  EXPECT_THAT(min_max.value().second, 4.0f);
}

}  // namespace
}  // namespace calibrator
}  // namespace tensorflow
