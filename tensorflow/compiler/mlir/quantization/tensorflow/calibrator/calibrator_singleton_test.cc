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
  CalibratorSingleton::ReportMinMax(/*id=*/"1", /*min_val=*/1.0f,
                                    /*max_val=*/2.0f);
  std::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax(/*id=*/"1");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 1.0f);
  EXPECT_EQ(min_max.value().second, 2.0f);

  CalibratorSingleton::ReportMinMax(/*id=*/"1", /*min_val=*/-1.0f,
                                    /*max_val=*/3.0f);
  min_max = CalibratorSingleton::GetMinMax(/*id=*/"1");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, -1.0f);
  EXPECT_EQ(min_max.value().second, 3.0f);

  CalibratorSingleton::ReportMinMax(/*id=*/"1", /*min_val=*/3.0f,
                                    /*max_val=*/5.0f);

  min_max = CalibratorSingleton::GetMinMax(/*id=*/"1");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, -1.0f);
  EXPECT_EQ(min_max.value().second, 5.0f);
}

TEST(CalibratorSingletonTest, DifferentSessions) {
  CalibratorSingleton::ReportMinMax(/*id=*/"2", /*min_val=*/1.0f,
                                    /*max_val=*/2.0f);
  std::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax(/*id=*/"2");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 1.0f);
  EXPECT_EQ(min_max.value().second, 2.0f);

  CalibratorSingleton::ReportMinMax(/*id=*/"3", /*min_val=*/-1.0f,
                                    /*max_val=*/3.0f);
  min_max = CalibratorSingleton::GetMinMax(/*id=*/"3");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, -1.0f);
  EXPECT_EQ(min_max.value().second, 3.0f);

  CalibratorSingleton::ReportMinMax(/*id=*/"2", /*min_val=*/3.0f,
                                    /*max_val=*/5.0f);
  min_max = CalibratorSingleton::GetMinMax(/*id=*/"2");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 1.0f);
  EXPECT_EQ(min_max.value().second, 5.0f);

  min_max = CalibratorSingleton::GetMinMax(/*id=*/"3");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, -1.0f);
  EXPECT_EQ(min_max.value().second, 3.0f);
}

TEST(CalibratorSingletonTest, ClearAndGetEmptyResult) {
  CalibratorSingleton::ReportMinMax(/*id=*/"4", /*min_val=*/1.0f,
                                    /*max_val=*/2.0f);
  std::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax(/*id=*/"4");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 1.0f);
  EXPECT_EQ(min_max.value().second, 2.0f);

  CalibratorSingleton::ClearCollectedInformation();

  min_max = CalibratorSingleton::GetMinMax(/*id=*/"4");
  EXPECT_FALSE(min_max.has_value());
}

TEST(CalibratorSingletonTest, ClearDataAndGetResults) {
  CalibratorSingleton::ReportMinMax(/*id=*/"5", /*min_val=*/1.0f,
                                    /*max_val=*/2.0f);
  std::optional<std::pair<float, float>> min_max =
      CalibratorSingleton::GetMinMax(/*id=*/"5");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 1.0f);
  EXPECT_EQ(min_max.value().second, 2.0f);

  CalibratorSingleton::ReportMinMax(/*id=*/"6", /*min_val=*/3.0f,
                                    /*max_val=*/4.0f);
  min_max = CalibratorSingleton::GetMinMax(/*id=*/"6");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 3.0f);
  EXPECT_EQ(min_max.value().second, 4.0f);

  CalibratorSingleton::ClearData(/*id=*/"5");

  min_max = CalibratorSingleton::GetMinMax(/*id=*/"5");
  EXPECT_FALSE(min_max.has_value());

  min_max = CalibratorSingleton::GetMinMax(/*id=*/"6");
  EXPECT_TRUE(min_max.has_value());
  EXPECT_EQ(min_max.value().first, 3.0f);
  EXPECT_EQ(min_max.value().second, 4.0f);
}

}  // namespace
}  // namespace calibrator
}  // namespace tensorflow
