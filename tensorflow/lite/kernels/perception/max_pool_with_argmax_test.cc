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

#include <cstdint>
#include <memory>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class MaxpoolingWithArgMaxOpModel : public SingleOpModel {
 public:
  MaxpoolingWithArgMaxOpModel(const TensorData& input, int stride_height,
                              int stride_width, int filter_height,
                              int filter_width, TfLitePadding padding,
                              const TensorData& output,
                              const TensorData& indices) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    indices_ = AddOutput(indices);

    std::vector<uint8_t> custom_option = CreateCustomOptions(
        stride_height, stride_width, filter_height, filter_width, padding);
    SetCustomOp("MaxPoolWithArgmax", custom_option, RegisterMaxPoolWithArgmax);
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<int32_t> GetIndices() { return ExtractVector<int32_t>(indices_); }

  std::vector<int> GetIndicesShape() { return GetTensorShape(indices_); }

 protected:
  int input_;
  int output_;
  int indices_;

 private:
  std::vector<uint8_t> CreateCustomOptions(int stride_height, int stride_width,
                                           int filter_height, int filter_width,
                                           TfLitePadding padding) {
    auto flex_builder = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_builder->StartMap();
    flex_builder->Bool("include_batch_in_index", false);
    if (padding == kTfLitePaddingValid) {
      flex_builder->String("padding", "VALID");
    } else {
      flex_builder->String("padding", "SAME");
    }

    auto start = flex_builder->StartVector("ksize");
    flex_builder->Add(1);
    flex_builder->Add(filter_height);
    flex_builder->Add(filter_width);
    flex_builder->Add(1);
    flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);

    auto strides_start = flex_builder->StartVector("strides");
    flex_builder->Add(1);
    flex_builder->Add(stride_height);
    flex_builder->Add(stride_width);
    flex_builder->Add(1);
    flex_builder->EndVector(strides_start, /*typed=*/true, /*fixed=*/false);

    flex_builder->EndMap(map_start);
    flex_builder->Finish();
    return flex_builder->GetBuffer();
  }
};

TEST(MaxpoolWithArgMaxTest, UnsupportedInt64Test) {
  EXPECT_DEATH_IF_SUPPORTED(MaxpoolingWithArgMaxOpModel model(
                                /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                                /*stride_height=*/2, /*stride_width=*/2,
                                /*filter_height=*/2, /*filter_width=*/2,
                                /*padding=*/kTfLitePaddingSame,
                                /*output=*/{TensorType_FLOAT32, {}},
                                /*indices=*/{TensorType_INT64, {}});
                            , "indices->type == kTfLiteInt32 was not true.");
}

TEST(MaxpoolWithArgMaxTest, SimpleTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});
  model.SetInput({0, 13, 2, 0, 0, 1, 4, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 2, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({13, 4}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 1, 2, 1}));
  EXPECT_THAT(model.GetIndices(), ElementsAreArray({1, 6}));
}

TEST(MaxpoolWithArgMaxTest, Strides2x1Test) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 2, 2}},
      /*stride_height=*/2, /*stride_width=*/1,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0, 0, 8});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({3, 4, 0, 4, 7, 8, 0, 8}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 2, 2}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({4, 7, 2, 7, 12, 15, 10, 15}));
}

TEST(MaxpoolWithArgMaxTest, Strides2x2Test) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 8, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0,
                  0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 8});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 3, 4, 0, 0, 7, 6, 8}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({0, 10, 13, 6, 16, 27, 20, 31}));
}

TEST(MaxpoolWithArgMaxTest, Strides2x2UnfitTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 7, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4,
                  0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 3, 2, 4, 0, 0, 5, 7}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({0, 10, 5, 13, 14, 16, 19, 27}));
}

TEST(MaxpoolWithArgMaxTest, PaddingValidTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 5, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/3,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput(
      {0, 0, 0, 0, 0, 0, 7, 0, 0, 10, 0, 0, 0, 0, 0, 0, 20, 0, 0, 19});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({7, 10, 20, 19}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetIndices(), ElementsAreArray({6, 9, 16, 19}));
}

TEST(MaxpoolWithArgMaxTest, PaddingValidUnfitTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {1, 4, 6, 1}},
      /*stride_height=*/2, /*stride_width=*/2,
      /*filter_height=*/2, /*filter_width=*/3,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({0, 0, 0, 0, 0,  0, 7, 0,  0,  10, 0, 0,
                  0, 0, 0, 0, 20, 0, 0, 19, 24, 1,  2, 44});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({7, 10, 24, 24}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(model.GetIndices(), ElementsAreArray({6, 9, 20, 20}));
}

TEST(MaxpoolWithArgMaxTest, InputWithBatchTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {2, 4, 12, 2}},
      /*stride_height=*/2, /*stride_width=*/3,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingSame,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({0,  0,  1,  0,  0,  0,  0,  0,  3,  4, 0,  0,  5, 0, 0,  6,
                  0,  0,  0,  0,  0,  0,  0,  2,  0,  0, 0,  0,  0, 0, 0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  7,  0, 0,  8,  9, 0, 0,  10,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0, 0, 15, 0,
                  0,  16, 0,  0,  0,  0,  0,  0,  11, 0, 0,  12, 0, 0, 0,  14,
                  13, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0, 0, 0,  0,
                  17, 18, 0,  0,  0,  30, 0,  20, 0,  0, 0,  0,  0, 0, 21, 0,
                  0,  0,  0,  0,  0,  24, 0,  0,  0,  0, 0,  0,  0, 0, 19, 0,
                  0,  0,  0,  22, 0,  0,  0,  0,  0,  0, 23, 0,  0, 0, 0,  0,
                  0,  0,  27, 28, 0,  0,  0,  0,  29, 0, 0,  0,  0, 0, 0,  32,
                  0,  0,  0,  0,  25, 26, 0,  0,  0,  0, 0,  0,  0, 0, 0,  0,
                  0,  0,  0,  0,  0,  0,  31, 0,  0,  0, 0,  0,  0, 0, 0,  0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  0,  3,  4,  5,  6,  9,  8,  11, 12, 13,
                                14, 15, 0,  0,  0,  17, 18, 19, 20, 21, 0,
                                23, 24, 27, 28, 29, 0,  31, 32, 25, 26}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({2,  1,  8,  9,  12, 15, 44, 43, 72, 75, 80,
                                79, 62, 61, 66, 67, 0,  1,  30, 7,  14, 13,
                                42, 21, 50, 51, 56, 55, 86, 63, 68, 69}));
}

TEST(MaxpoolWithArgMaxTest, InputWithBatchAndPaddingValidTest) {
  MaxpoolingWithArgMaxOpModel model(
      /*input=*/{TensorType_FLOAT32, {2, 4, 11, 2}},
      /*stride_height=*/2, /*stride_width=*/3,
      /*filter_height=*/2, /*filter_width=*/2,
      /*padding=*/kTfLitePaddingValid,
      /*output=*/{TensorType_FLOAT32, {}},
      /*indices=*/{TensorType_INT32, {}});

  model.SetInput({0,  0,  1,  0, 0, 0, 0,  0,  3,  4,  0,  0,  5,  0,  0,  6,
                  0,  0,  0,  0, 0, 0, 0,  2,  0,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0, 0, 0, 0,  0,  7,  0,  0,  8,  9,  0,  0,  10,
                  0,  0,  0,  0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  15, 0,
                  0,  16, 0,  0, 0, 0, 0,  0,  11, 0,  0,  12, 0,  0,  0,  14,
                  13, 0,  0,  0, 0, 0, 0,  0,  17, 18, 0,  0,  0,  30, 0,  20,
                  0,  0,  0,  0, 0, 0, 21, 0,  0,  0,  0,  0,  0,  24, 0,  0,
                  0,  0,  0,  0, 0, 0, 19, 0,  0,  0,  0,  22, 0,  0,  0,  0,
                  0,  0,  23, 0, 0, 0, 0,  0,  0,  0,  27, 28, 0,  0,  0,  0,
                  29, 0,  0,  0, 0, 0, 0,  32, 0,  0,  0,  0,  25, 26, 0,  0,
                  0,  0,  0,  0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  31, 0});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 29, 0,  31, 32}));
  EXPECT_THAT(model.GetIndicesShape(), ElementsAreArray({2, 2, 4, 2}));
  EXPECT_THAT(model.GetIndices(),
              ElementsAreArray({2,  23, 8,  9,  12, 15, 40, 43, 44, 47, 72,
                                75, 80, 79, 62, 65, 0,  1,  30, 7,  14, 35,
                                42, 21, 68, 69, 50, 51, 56, 57, 86, 63}));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
