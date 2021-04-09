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

#include <initializer_list>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class DepthToSpaceOpModel : public SingleOpModel {
 public:
  DepthToSpaceOpModel(const TensorData& tensor_data, int block_size) {
    input_ = AddInput(tensor_data);
    output_ = AddOutput(tensor_data);
    SetBuiltinOp(BuiltinOperator_DEPTH_TO_SPACE,
                 BuiltinOptions_DepthToSpaceOptions,
                 CreateDepthToSpaceOptions(builder_, block_size).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }
  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(DepthToSpaceOpModel, BadBlockSize) {
  EXPECT_DEATH(DepthToSpaceOpModel({TensorType_FLOAT32, {1, 1, 1, 4}}, 4),
               "Cannot allocate tensors");
}
#endif

TEST(DepthToSpaceOpModel, Float32) {
  DepthToSpaceOpModel m({TensorType_FLOAT32, {1, 1, 1, 4}}, 2);
  m.SetInput<float>({1.4, 2.3, 3.2, 4.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.4, 2.3, 3.2, 4.1}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 2, 1));
}

TEST(DepthToSpaceOpModel, Uint8) {
  DepthToSpaceOpModel m({TensorType_UINT8, {1, 1, 2, 4}}, 2);
  m.SetInput<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({1, 2, 5, 6, 3, 4, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 2, 4, 1));
}

TEST(DepthToSpaceOpModel, int8) {
  DepthToSpaceOpModel m({TensorType_INT8, {1, 2, 1, 4}}, 2);
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 2, 1));
}

TEST(DepthToSpaceOpModel, Int32) {
  DepthToSpaceOpModel m({TensorType_INT32, {1, 2, 2, 4}}, 2);
  m.SetInput<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray(
                  {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 4, 4, 1));
}

TEST(DepthToSpaceOpModel, Int64) {
  DepthToSpaceOpModel m({TensorType_INT64, {1, 1, 1, 1}}, 1);
  m.SetInput<int64_t>({4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({4}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(1, 1, 1, 1));
}

}  // namespace
}  // namespace tflite
