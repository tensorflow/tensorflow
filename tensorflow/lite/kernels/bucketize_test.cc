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
#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

template <typename T>
class BucketizeOpModel : public SingleOpModel {
 public:
  BucketizeOpModel(const TensorData& input,
                   const std::vector<float>& boundaries) {
    input_ = AddInput(input);
    boundaries_ = boundaries;

    output_ = AddOutput({TensorType_INT32, input.shape});

    SetBuiltinOp(BuiltinOperator_BUCKETIZE, BuiltinOptions_BucketizeOptions,
                 CreateBucketizeOptions(
                     builder_, builder_.CreateVector<float>(boundaries_))
                     .Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  const std::vector<float>& boundaries() { return boundaries_; }

  std::vector<int> GetOutput() { return ExtractVector<int>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  std::vector<float> boundaries_;
  int output_;
};

TEST(BucketizeOpTest, Float) {
  // Buckets are: (-inf, 0.), [0., 10.), [10., 100.), [100., +inf).
  BucketizeOpModel<float> model(
      /*input=*/{/*type=*/TensorType_FLOAT32, /*shape=*/{3, 2}},
      /*boundaries=*/{0.0f, 10.0f, 100.0f});

  // input: [[-5, 10000], [150, 10], [5, 100]]
  model.PopulateTensor<float>(model.input(),
                              {-5.0f, 10000.0f, 150.0f, 10.0f, 5.0f, 100.0f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  // output: [[0, 3], [3, 2], [1, 3]]
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 3, 3, 2, 1, 3}));
}

TEST(BucketizeOpTest, Int32) {
  // Buckets are: (-inf, 0.), [0., 10.), [10., 100.), [100., +inf).
  BucketizeOpModel<int32_t> model(
      /*input=*/{/*type=*/TensorType_INT32, /*shape=*/{3, 2}},
      /*boundaries=*/{0, 10, 100});

  // input: [[-5, 10000], [150, 10], [5, 100]]
  model.PopulateTensor<int32_t>(model.input(), {-5, 10000, 150, 10, 5, 100});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  // output: [[0, 3], [3, 2], [1, 3]]
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3, 2));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0, 3, 3, 2, 1, 3}));
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(BucketizeOpTest, UnsortedBuckets) {
  EXPECT_DEATH(BucketizeOpModel<float>(
                   /*input=*/{/*type=*/TensorType_INT32, /*shape=*/{3, 2}},
                   /*boundaries=*/{0, 10, -10}),
               "Expected sorted boundaries");
}
#endif

}  // namespace
}  // namespace tflite
