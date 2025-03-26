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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
using testing::ElementsAreArray;

class SpaceToDepthOpModel : public SingleOpModelWithHexagon {
 public:
  SpaceToDepthOpModel(const TensorData& tensor_data, int block_size,
                      BuiltinOperator type) {
    input_ = AddInput(tensor_data);
    output_ = AddOutput(tensor_data);
    if (type == BuiltinOperator_SPACE_TO_DEPTH) {
      SetBuiltinOp(BuiltinOperator_SPACE_TO_DEPTH,
                   BuiltinOptions_SpaceToDepthOptions,
                   CreateSpaceToDepthOptions(builder_, block_size).Union());
    } else {
      SetBuiltinOp(BuiltinOperator_DEPTH_TO_SPACE,
                   BuiltinOptions_DepthToSpaceOptions,
                   CreateDepthToSpaceOptions(builder_, block_size).Union());
    }
    BuildInterpreter({GetShape(input_)});
  }

  template <typename integer_type>
  void SetInput(const std::vector<integer_type>& data) {
    PopulateTensor<integer_type>(input_, data);
  }

  template <typename integer_type>
  std::vector<integer_type> GetOutput() {
    return ExtractVector<integer_type>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(SpaceToDepthOpModel, SpaceToDepth_UInt8) {
  SpaceToDepthOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -5, 5}, 2,
                        BuiltinOperator_SPACE_TO_DEPTH);
  m.SetInput<uint8_t>({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SpaceToDepthOpModel, SpaceToDepth_Int8) {
  SpaceToDepthOpModel m({TensorType_INT8, {1, 2, 2, 1}, -5, 5}, 2,
                        BuiltinOperator_SPACE_TO_DEPTH);
  m.SetInput<int8_t>({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SpaceToDepthOpModel, DepthToSpace_UInt8) {
  SpaceToDepthOpModel m({TensorType_UINT8, {1, 1, 2, 4}, -8, 8}, 2,
                        BuiltinOperator_DEPTH_TO_SPACE);
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
}

TEST(SpaceToDepthOpModel, DepthToSpace_Int8) {
  SpaceToDepthOpModel m({TensorType_INT8, {1, 1, 2, 4}, -8, 8}, 2,
                        BuiltinOperator_DEPTH_TO_SPACE);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
}

}  // namespace tflite
