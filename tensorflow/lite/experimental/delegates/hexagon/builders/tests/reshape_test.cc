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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using ::testing::ElementsAreArray;

// There are three ways to specify the output shape of a Reshape
// op.
enum class ShapeSpecificationType {
  // The output shape is hardcoded in the ReshapeOptions object.
  kAsReshapeOption,
  // The output shape is specified as an input tensor, which is connected to a
  // Const node, which is guaranteed not to change once inference starts. The
  // shape is also hardcoded as in kAsReshapeOption.
  kAsConstantTensor,
  // The output shape is specified as an input tensor that can change based on
  // external input. That is, the shape is not know before the inference
  // starts. The shape is also hardcoded as in kAsReshapeOption.
  // TODO(b/152562126): Enable this when Hexagon supports dynamic sizes.
  kAsTensor,
};

template <typename T>
class ReshapeOpModel : public SingleOpModelWithHexagon {
 public:
  ReshapeOpModel(std::initializer_list<int> input_shape,
                 std::initializer_list<int> shape_shape,
                 std::initializer_list<int> shape_data,
                 ShapeSpecificationType shape_type) {
    switch (shape_type) {
      case ShapeSpecificationType::kAsTensor:
        BuildWithTensorShape(input_shape, shape_shape, shape_data);
        break;
      case ShapeSpecificationType::kAsConstantTensor:
        BuildWithConstantTensorShape(input_shape, shape_shape, shape_data);
        break;
      case ShapeSpecificationType::kAsReshapeOption:
        // In this case the shape of the new shape doesn't matter. It is
        // always hardcoded as a flat vector.
        BuildWithHardcodedShape(input_shape, shape_data);
        break;
    }
  }

  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  void SetStringInput(std::initializer_list<string> data) {
    PopulateStringTensor(input_, data);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  void BuildWithHardcodedShape(std::initializer_list<int> input_shape,
                               std::initializer_list<int> shape_data) {
    input_ = AddInput({GetTensorType<T>(), input_shape});
    output_ = AddOutput(GetTensorType<T>());
    SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(builder_, builder_.CreateVector<int>(shape_data))
            .Union());
    BuildInterpreter({GetShape(input_)});
  }

  void BuildWithTensorShape(std::initializer_list<int> input_shape,
                            std::initializer_list<int> shape_shape,
                            std::initializer_list<int> shape_data) {
    input_ = AddInput({GetTensorType<T>(), input_shape});
    output_ = AddOutput(GetTensorType<T>());
    int shape_input_tensor = AddInput({TensorType_INT32, shape_shape});
    // Note how shape also appears in ReshapeOptions
    SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(builder_, builder_.CreateVector<int>(shape_data))
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(shape_input_tensor)});
    if (shape_data.size() != 0) {
      PopulateTensor<int32_t>(shape_input_tensor, shape_data);
    }
  }

  void BuildWithConstantTensorShape(std::initializer_list<int> input_shape,
                                    std::initializer_list<int> shape_shape,
                                    std::initializer_list<int> shape_data) {
    input_ = AddInput({GetTensorType<T>(), input_shape});
    output_ = AddOutput(GetTensorType<T>());
    AddConstInput(TensorType_INT32, shape_data, shape_shape);
    // Note how the shape also appears in the ReshapeOptions.
    SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(builder_, builder_.CreateVector<int>(shape_data))
            .Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input_;
  int output_;
};

template <typename T>
class ReshapeOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<uint8_t, int8_t>;
TYPED_TEST_SUITE(ReshapeOpTest, DataTypes);

TYPED_TEST(ReshapeOpTest, RegularShapes) {
  std::vector<ShapeSpecificationType> shape_types = {
      ShapeSpecificationType::kAsReshapeOption,
      ShapeSpecificationType::kAsConstantTensor};

  for (ShapeSpecificationType shape_type : shape_types) {
    ReshapeOpModel<TypeParam> m({1, 2, 4, 1}, {3}, {2, 2, 2}, shape_type);
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
    ReshapeOpModel<TypeParam> m({1, 2, 4, 1}, {3}, {2, 1, -1}, shape_type);
    m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
    m.ApplyDelegateAndInvoke();
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 4}));
  }
}

}  // namespace tflite
