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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

enum class Agreement { kBroadcast, kExtend, kBroadcastNotExtend, kNeither };

// A pair of Shapes and whether they should agree up to broadcasting, extending
// or neither.
struct ShapePair {
  Shape left;
  Shape right;
  Agreement agreement;
};

std::vector<ShapePair> CreateShapePairs() {
  return std::vector<ShapePair>(
      {// These agree up to broadcast.
       {Shape({3}), Shape({3}), Agreement::kBroadcast},
       {Shape({256, 256, 3}), Shape({256, 256, 3}), Agreement::kBroadcast},
       {Shape({256, 256, 3}), Shape({3}), Agreement::kBroadcast},
       {Shape({8, 1, 6, 1}), Shape({7, 1, 5}), Agreement::kBroadcast},
       {Shape({}), Shape({3}), Agreement::kBroadcast},
       {Shape({}), Shape({3, 1}), Agreement::kBroadcast},

       // These extend (and therefore broadcast).
       {Shape({3}), Shape({3}), Agreement::kExtend},
       {Shape({256, 256, 3}), Shape({256, 256, 3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({1, 1, 3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({1, 3}), Agreement::kExtend},

       // These strictly broadcast and do not extend.
       {Shape({256, 256, 3}), Shape({3}), Agreement::kBroadcastNotExtend},
       {Shape({5, 4}), Shape({1}), Agreement::kBroadcastNotExtend},
       {Shape({5, 4}), Shape({4}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({15, 1, 5}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({3, 5}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({3, 1}), Agreement::kBroadcastNotExtend},
       {Shape({3, 1}), Shape({}), Agreement::kBroadcastNotExtend},

       // These do not broadcast (and therefore also do not extend).
       {Shape({3}), Shape({4}), Agreement::kNeither},
       {Shape({2, 1}), Shape({8, 4, 3}), Agreement::kNeither}});
}

// ShapeTest is an empty parameterized test fixture since there is no state.
class ShapeTest : public ::testing::TestWithParam<ShapePair> {};

TEST_P(ShapeTest, Agrees) {
  const ShapePair& param = GetParam();

  switch (param.agreement) {
    case Agreement::kBroadcast: {
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
    case Agreement::kExtend: {
      EXPECT_TRUE(ShapesAgreeUpToExtending(param.left, param.right));
      // Anything that extends should also broadcast.
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
    case Agreement::kBroadcastNotExtend: {
      // Verify that it strictly broadcasts but does not extend.
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      EXPECT_FALSE(ShapesAgreeUpToExtending(param.left, param.right));
      break;
    }
    case Agreement::kNeither: {
      EXPECT_FALSE(ShapesAgreeUpToExtending(param.left, param.right));
      EXPECT_FALSE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AgreeBroadcast, ShapeTest,
                         ::testing::ValuesIn(CreateShapePairs()));

static const char kNegativeValuesMessage[] =
    "Tensor shape should not include negative values";
static const char kLargeTensorMessage[] = "Tensor shape is too large";

TEST(NumElementsTest, Int) {
  int count;
  absl::Status status = absl::OkStatus();

  status = NumElements(std::vector<int>{1024, 1024, 2047}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 2146435072);

  status = NumElements(std::vector<int>{1024, 0, 2048}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 0);

  status = NumElements(std::vector<int>{1, 2, -3}, &count);
  EXPECT_EQ(status.message(), kNegativeValuesMessage);

  status = NumElements(std::vector<int>{1024, 1024, 2048}, &count);
  EXPECT_EQ(status.message(), kLargeTensorMessage);
}

TEST(NumElementsTest, Int32) {
  int32_t count;
  absl::Status status = absl::OkStatus();

  status = NumElements(std::vector<int32_t>{1024, 1024, 2047}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 2146435072);

  status = NumElements(std::vector<int32_t>{1, 2, -3}, &count);
  EXPECT_EQ(status.message(), kNegativeValuesMessage);

  status = NumElements(std::vector<int32_t>{1024, 1024, 2048}, &count);
  EXPECT_EQ(status.message(), kLargeTensorMessage);
}

TEST(NumElementsTest, Int64) {
  int64_t count;
  absl::Status status = absl::OkStatus();

  status = NumElements(std::vector<int64_t>{16777216, 16777216, 32767}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 9223090561878065152LL);

  status = NumElements(std::vector<int64_t>{1, 2, -3}, &count);
  EXPECT_EQ(status.message(), kNegativeValuesMessage);

  status = NumElements(std::vector<int64_t>{16777216, 16777216, 32768}, &count);
  EXPECT_EQ(status.message(), kLargeTensorMessage);
}

TEST(NumElementsTest, UnsignedInt32) {
  uint32_t count;
  absl::Status status = absl::OkStatus();

  status = NumElements(std::vector<uint32_t>{1024, 2048, 2047}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 4292870144);

  status = NumElements(std::vector<int>{1, 2, -3}, &count);
  EXPECT_EQ(status.message(), kNegativeValuesMessage);

  status = NumElements(std::vector<uint32_t>{1024, 2048, 2048}, &count);
  EXPECT_EQ(status.message(), kLargeTensorMessage);
}

TEST(NumElementsTest, UnsignedInt64) {
  uint64_t count;
  absl::Status status = absl::OkStatus();

  status =
      NumElements(std::vector<uint64_t>{16777216, 16777216, 65535}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 18446462598732840960ULL);

  status = NumElements(std::vector<int>{1, 2, -3}, &count);
  EXPECT_EQ(status.message(), kNegativeValuesMessage);

  status =
      NumElements(std::vector<uint64_t>{16777216, 16777216, 65536}, &count);
  EXPECT_EQ(status.message(), kLargeTensorMessage);
}

TEST(NumElementsTest, Scalar) {
  absl::Status status = absl::OkStatus();

  int32_t count;
  status = NumElements(std::vector<int32_t>{}, &count);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(count, 1);

  uint64_t countu64;
  status = NumElements(std::vector<uint64_t>{}, &countu64);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(countu64, 1ULL);
}

TEST(FusedActivationTest, DefaultsToUnfused) {
  EXPECT_TRUE(OperatorSupportsFusedActivation(OperatorType::kAdd));
  EXPECT_FALSE(OperatorSupportsFusedActivation(OperatorType::kNone));
  EXPECT_FALSE(OperatorSupportsFusedActivation(static_cast<OperatorType>(255)));
}

}  // namespace toco

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
