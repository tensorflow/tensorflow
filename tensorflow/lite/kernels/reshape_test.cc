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
#include <stdint.h>

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/reshape_test_common.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

template <typename T>
class ReshapeOpTest : public ::testing::Test {
 public:
  static std::vector<ShapeSpecificationType> _range_;
};

template <>
std::vector<ShapeSpecificationType>
    ReshapeOpTest<ShapeSpecificationType>::_range_{
        ShapeSpecificationType::kAsReshapeOption,
        ShapeSpecificationType::kAsConstantTensor,
        ShapeSpecificationType::kAsTensor};

using DataTypes = ::testing::Types<float, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(ReshapeOpTest, DataTypes);

TYPED_TEST(ReshapeOpTest, MismatchedDimensions) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    if (shape_type == ShapeSpecificationType::kAsTensor) {
      ReshapeOpModel<TypeParam> m({1, 2, 4, 1}, {2}, {2, 1}, shape_type);
      m.SetInput({3});
      EXPECT_NE(m.Invoke(), kTfLiteOk)
          << "num_input_elements != num_output_elements";
    } else {
#ifdef GTEST_HAS_DEATH_TEST
      EXPECT_DEATH(
          ReshapeOpModel<TypeParam>({1, 2, 4, 1}, {2}, {2, 1}, shape_type),
          "num_input_elements != num_output_elements");
#endif
    }
  }
}

TYPED_TEST(ReshapeOpTest, TooManyDimensions) {
#ifdef GTEST_HAS_DEATH_TEST
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    EXPECT_DEATH(
        ReshapeOpModel<TypeParam>({1, 1, 2, 1, 1, 1, 1, 1, 1}, {9},
                                  {1, 1, 1, 1, 1, 1, 1, 1, 2}, shape_type),
        "Found too many dimensions");
  }
#endif
}

TYPED_TEST(ReshapeOpTest, TooManySpecialDimensions) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    if (shape_type != ShapeSpecificationType::kAsTensor) {
#ifdef GTEST_HAS_DEATH_TEST
      EXPECT_DEATH(ReshapeOpModel<TypeParam>({1, 2, 4, 1}, {4}, {-1, -1, 2, 4},
                                             shape_type),
                   "stretch_dim != -1");
#endif
    } else {
      ReshapeOpModel<TypeParam> m({1, 2, 4, 1}, {4}, {-1, -1, 2, 4},
                                  shape_type);
      EXPECT_NE(m.Invoke(), kTfLiteOk) << "stretch_dim != -1";
    }
  }
}

// Create the model with a 2x2 shape. Processing still works because the new
// shape ends up being hardcoded as a flat vector.
TYPED_TEST(ReshapeOpTest, InvalidShape) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    if (SingleOpModel::GetForceUseNnapi() &&
        shape_type == ShapeSpecificationType::kAsTensor) {
      // NNAPI delegate does not support RESHAPE with shape as a non-constant
      // tensor.
      continue;
    }
    ReshapeOpModel<TypeParam> m({1, 2, 2}, {2, 2}, {1, 2, 2, 1}, shape_type);
    m.SetInput({5, 6, 7, 8});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6, 7, 8}));
  }
}

// This is the normal scenario, where shape is a vector.
TYPED_TEST(ReshapeOpTest, RegularShapes) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    if (SingleOpModel::GetForceUseNnapi() &&
        shape_type == ShapeSpecificationType::kAsTensor) {
      // NNAPI delegate does not support RESHAPE with shape as a non-constant
      // tensor.
      continue;
    }
    ReshapeOpModel<TypeParam> m({1, 2, 4, 1}, {3}, {2, 2, 2}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
  }
}

TYPED_TEST(ReshapeOpTest, WithStretchDimension) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    if (SingleOpModel::GetForceUseNnapi() &&
        shape_type == ShapeSpecificationType::kAsTensor) {
      // NNAPI delegate does not support RESHAPE with shape as a non-constant
      // tensor.
      continue;
    }
    ReshapeOpModel<TypeParam> m({1, 2, 4, 1}, {3}, {2, 1, -1}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 4}));
  }
}

// Shape is specified as '[]', which is the modern way to represent scalar
// input and output.
TYPED_TEST(ReshapeOpTest, ScalarOutput) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    ReshapeOpModel<TypeParam> m({1}, {0}, {}, shape_type);
    m.SetInput({3});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
    EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  }
}

TYPED_TEST(ReshapeOpTest, ZeroInShape) {
  if (SingleOpModel::GetForceUseNnapi()) {
    return;
  }
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    ReshapeOpModel<TypeParam> m({4, 0}, {3}, {2, 0, -1}, shape_type);
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 0, 2}));
  }
}

// Some old models specify '[0]' as the new shape, indicating that both input
// and output are scalars.
TYPED_TEST(ReshapeOpTest, LegacyScalarOutput) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    if (shape_type == ShapeSpecificationType::kAsConstantTensor) {
#ifdef GTEST_HAS_DEATH_TEST
      EXPECT_DEATH(ReshapeOpModel<TypeParam>({1}, {1}, {0}, shape_type),
                   "num_input_elements != num_output_elements");
#endif
    } else if (shape_type == ShapeSpecificationType::kAsTensor) {
      ReshapeOpModel<TypeParam> m({1}, {1}, {0}, shape_type);
      m.SetInput({3});
      ASSERT_NE(m.Invoke(), kTfLiteOk)
          << "num_input_elements != num_output_elements";
    } else {
      ReshapeOpModel<TypeParam> m({1}, {1}, {0}, shape_type);
      m.SetInput({3});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
      EXPECT_THAT(m.GetOutputShape(), IsEmpty());
    }
  }
}

TYPED_TEST(ReshapeOpTest, Strings) {
  for (ShapeSpecificationType shape_type :
       ReshapeOpTest<ShapeSpecificationType>::_range_) {
    ReshapeOpModel<string> m({1, 2, 4, 1}, {3}, {2, 2, 2}, shape_type);
    m.SetStringInput({"1", "2", "3", "4", "5", "6", "7", "8"});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({"1", "2", "3", "4", "5", "6", "7", "8"}));
  }
}
}  // namespace
}  // namespace tflite
