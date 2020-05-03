/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/reshape_test_common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using ::testing::ElementsAreArray;

template <typename T>
class ReshapeOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<uint8_t, int8_t>;
TYPED_TEST_SUITE(ReshapeOpTest, DataTypes);

TYPED_TEST(ReshapeOpTest, RegularShapes) {
  std::vector<ShapeSpecificationType> shape_types = {
      ShapeSpecificationType::kAsReshapeOption,
      ShapeSpecificationType::kAsConstantTensor};

  for (ShapeSpecificationType shape_type : shape_types) {
    ReshapeOpModel<TypeParam, SingleOpModelWithHexagon> m(
        {1, 2, 4, 1}, {3}, {2, 2, 2}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
  }
}

TYPED_TEST(ReshapeOpTest, WithStretchDimension) {
  std::vector<ShapeSpecificationType> shape_types = {
      ShapeSpecificationType::kAsReshapeOption,
      ShapeSpecificationType::kAsConstantTensor};

  for (ShapeSpecificationType shape_type : shape_types) {
    ReshapeOpModel<TypeParam, SingleOpModelWithHexagon> m(
        {1, 2, 4, 1}, {3}, {2, 1, -1}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 4}));
  }
}

}  // namespace tflite
