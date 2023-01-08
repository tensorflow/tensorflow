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

#include <initializer_list>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {

using ::testing::ElementsAreArray;

enum PoolType {
  kAverage,
  kMax,
};

template <typename T>
class BasePoolingOpModel : public SingleOpModel {
 public:
  BasePoolingOpModel(PoolType pool_type, TensorData input, int filter_d,
                     int filter_h, int filter_w, TensorData output,
                     TfLitePadding padding = kTfLitePaddingValid,
                     int stride_d = 2, int stride_h = 2, int stride_w = 2) {
    if (input.type == TensorType_FLOAT32) {
      // Clear quantization params.
      input.min = input.max = 0.f;
      output.min = output.max = 0.f;
    }
    input_ = AddInput(input);
    output_ = AddOutput(output);

    std::vector<uint8_t> custom_option = CreateCustomOptions(
        stride_d, stride_h, stride_w, filter_d, filter_h, filter_w, padding);
    if (pool_type == kAverage) {
      SetCustomOp("AveragePool3D", custom_option,
                  ops::custom::Register_AVG_POOL_3D);
    } else {
      SetCustomOp("MaxPool3D", custom_option,
                  ops::custom::Register_MAX_POOL_3D);
    }
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  std::vector<float> GetOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;

 private:
  std::vector<uint8_t> CreateCustomOptions(int stride_depth, int stride_height,
                                           int stride_width, int filter_depth,
                                           int filter_height, int filter_width,
                                           TfLitePadding padding) {
    auto flex_builder = std::make_unique<flexbuffers::Builder>();
    size_t map_start = flex_builder->StartMap();
    flex_builder->String("data_format", "NDHWC");
    if (padding == kTfLitePaddingValid) {
      flex_builder->String("padding", "VALID");
    } else {
      flex_builder->String("padding", "SAME");
    }

    auto start = flex_builder->StartVector("ksize");
    flex_builder->Add(1);
    flex_builder->Add(filter_depth);
    flex_builder->Add(filter_height);
    flex_builder->Add(filter_width);
    flex_builder->Add(1);
    flex_builder->EndVector(start, /*typed=*/true, /*fixed=*/false);

    auto strides_start = flex_builder->StartVector("strides");
    flex_builder->Add(1);
    flex_builder->Add(stride_depth);
    flex_builder->Add(stride_height);
    flex_builder->Add(stride_width);
    flex_builder->Add(1);
    flex_builder->EndVector(strides_start, /*typed=*/true, /*fixed=*/false);

    flex_builder->EndMap(map_start);
    flex_builder->Finish();
    return flex_builder->GetBuffer();
  }
};

template <>
void BasePoolingOpModel<float>::SetInput(const std::vector<float>& data) {
  PopulateTensor(input_, data);
}

template <>
std::vector<float> BasePoolingOpModel<float>::GetOutput() {
  return ExtractVector<float>(output_);
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(AveragePoolingOpTest, InvalidDimSize) {
  EXPECT_DEATH(BasePoolingOpModel<float> m(
                   kAverage,
                   /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                   /*filter_d=*/2,
                   /*filter_h=*/2, /*filter_w=*/2,
                   /*output=*/{TensorType_FLOAT32, {}},
                   /*padding=*/kTfLitePaddingValid, /*stride_d=*/1,
                   /*stride_h=*/1, /*stride_w=*/1),
               "NumDimensions.input. != 5 .4 != 5.");
}

TEST(AveragePoolingOpTest, ZeroStride) {
  EXPECT_DEATH(BasePoolingOpModel<float> m(
                   kAverage,
                   /*input=*/{TensorType_FLOAT32, {1, 2, 2, 4, 1}},
                   /*filter_d=*/2,
                   /*filter_h=*/2, /*filter_w=*/2,
                   /*output=*/{TensorType_FLOAT32, {}},
                   /*padding=*/kTfLitePaddingValid, /*stride_d=*/0,
                   /*stride_h=*/0, /*stride_w=*/0),
               "Cannot allocate tensors");
}
#endif

template <typename T>
class AveragePoolingOpTest : public ::testing::Test {};

template <typename T>
class MaxPoolingOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, int8_t, int16_t>;
TYPED_TEST_SUITE(AveragePoolingOpTest, DataTypes);
TYPED_TEST_SUITE(MaxPoolingOpTest, DataTypes);

TYPED_TEST(AveragePoolingOpTest, AveragePool) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.125, 4.25}));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolFilterH1) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/1, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.75, 5.75}));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolPaddingSameStride1) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingSame,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({2.875, 4.125, 4.5, 4.5, 3.0, 3.25, 3.25, 3.5,
                                2.5, 4.0, 5.75, 5.5, 2.5, 2.0, 3.0, 4.0}));
}

TYPED_TEST(AveragePoolingOpTest, AveragePoolPaddingValidStride1) {
  BasePoolingOpModel<TypeParam> m(
      kAverage,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingValid,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.875, 4.125, 4.5}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPool) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6.0, 10.0}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolFilterH1) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/1, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375});
  m.SetInput({0, 6, 2, 4, 4, 5, 1, 4, 3, 2, 10, 7, 2, 3, 5, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolPaddingSameStride1) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingSame,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10, 10, 7, 5, 5, 4, 4, 3, 10,
                                               10, 7, 3, 2, 4, 4}));
}

TYPED_TEST(MaxPoolingOpTest, MaxPoolPaddingValidStride1) {
  BasePoolingOpModel<TypeParam> m(
      kMax,
      /*input=*/{GetTensorType<TypeParam>(), {1, 2, 2, 4, 1}, 0, 15.9375},
      /*filter_d=*/2,
      /*filter_h=*/2, /*filter_w=*/2,
      /*output=*/{GetTensorType<TypeParam>(), {}, 0, 15.9375},
      kTfLitePaddingValid,
      /*stride_d=*/1, /*stride_h=*/1,
      /*stride_w=*/1);
  m.SetInput({0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6.0, 10.0, 10.0}));
}

}  // namespace tflite
