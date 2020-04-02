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

  void SetInput(const std::vector<uint8_t>& data) {
    PopulateTensor<uint8_t>(input_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(SpaceToDepthOpModel, SpaceToDepth) {
  SpaceToDepthOpModel m({TensorType_UINT8, {1, 2, 2, 1}, -5, 5}, 2,
                        BuiltinOperator_SPACE_TO_DEPTH);
  m.SetInput({1, 2, 3, 4});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SpaceToDepthOpModel, DepthToSpace) {
  SpaceToDepthOpModel m({TensorType_UINT8, {1, 1, 2, 4}, -8, 8}, 2,
                        BuiltinOperator_DEPTH_TO_SPACE);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
}

}  // namespace tflite
