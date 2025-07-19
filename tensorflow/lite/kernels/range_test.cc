/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "Eigen/Core"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

template <typename T>
class RangeOpModel : public SingleOpModel {
 public:
  explicit RangeOpModel(const TensorType& dtype) {
    start_ = AddInput(dtype);
    limit_ = AddInput(dtype);
    delta_ = AddInput(dtype);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  explicit RangeOpModel(const TensorType& dtype, const std::vector<T>& start,
                        const std::vector<T>& limit,
                        const std::vector<T>& delta) {
    start_ = AddConstInput(dtype, start);
    limit_ = AddConstInput(dtype, limit);
    delta_ = AddConstInput(dtype, delta);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  explicit RangeOpModel(const TensorData& start, const TensorData& limit,
                        const TensorData& delta) {
    start_ = AddInput(start);
    limit_ = AddInput(limit);
    delta_ = AddInput(delta);
    output_ = AddOutput(start.type);
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  int start() { return start_; }
  int limit() { return limit_; }
  int delta() { return delta_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
  int output() const { return output_; }

  template <typename output_type>
  std::vector<output_type> ExtractDequantVector(int index) {
    auto vec = ExtractVector<T>(index);
    TfLiteTensor* t = interpreter_->tensor(index);
    auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(t->quantization.params);
    float scaling_factor = affine_quantization->scale->data[0];
    int zero_point = affine_quantization->zero_point->data[0];
    std::vector<output_type> output;
    for (const auto& v : vec) {
      output.push_back(static_cast<output_type>(
          (static_cast<output_type>(v) - zero_point) * scaling_factor));
    }
    return output;
  }

 private:
  int start_;
  int limit_;
  int delta_;
  int output_;
};

// for quantized, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0 * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

TEST(RangeOpModel, Simple) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {0});
  model.PopulateTensor<int32_t>(model.limit(), {4});
  model.PopulateTensor<int32_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, SimpleConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {0}, {4}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, DeltaGreaterThanOne) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {2});
  model.PopulateTensor<int32_t>(model.limit(), {9});
  model.PopulateTensor<int32_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, DeltaGreaterThanOneConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, NegativeDelta) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {10});
  model.PopulateTensor<int32_t>(model.limit(), {3});
  model.PopulateTensor<int32_t>(model.delta(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, NegativeDeltaConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {10}, {3}, {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatSimple) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {0});
  model.PopulateTensor<float>(model.limit(), {4});
  model.PopulateTensor<float>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, FloatSimpleConst) {
  RangeOpModel<float> model(TensorType_FLOAT32, {0}, {4}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, FloatDeltaGreaterThanOne) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {2});
  model.PopulateTensor<float>(model.limit(), {9});
  model.PopulateTensor<float>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, FloatDeltaGreaterThanOneConst) {
  RangeOpModel<float> model(TensorType_FLOAT32, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, FloatNegativeDelta) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {10});
  model.PopulateTensor<float>(model.limit(), {3});
  model.PopulateTensor<float>(model.delta(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatNegativeDeltaConst) {
  RangeOpModel<float> model(TensorType_FLOAT32, {10}, {3}, {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, EmptyOutput) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {0});
  model.PopulateTensor<int32_t>(model.limit(), {0});
  model.PopulateTensor<int32_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, EmptyOutputConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {0}, {0}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, Int64Simple) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {0});
  model.PopulateTensor<int64_t>(model.limit(), {4});
  model.PopulateTensor<int64_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, Int64SimpleConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {0}, {4}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, Int64DeltaGreaterThanOne) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {2});
  model.PopulateTensor<int64_t>(model.limit(), {9});
  model.PopulateTensor<int64_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int64DeltaGreaterThanOneConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int64NegativeDelta) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {10});
  model.PopulateTensor<int64_t>(model.limit(), {3});
  model.PopulateTensor<int64_t>(model.delta(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, Int64NegativeDeltaConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {10}, {3}, {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, Int64EmptyOutput) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {0});
  model.PopulateTensor<int64_t>(model.limit(), {0});
  model.PopulateTensor<int64_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, Int64EmptyOutputConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {0}, {0}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, SimpleInt8) {
  RangeOpModel<int8_t> model(TensorType_INT8);
  model.PopulateTensor<int8_t>(model.start(), {0});
  model.PopulateTensor<int8_t>(model.limit(), {4});
  model.PopulateTensor<int8_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, SimpleInt8Quantized) {
  float kQuantizedTolerance = GetTolerance<int8_t>(-0.0f, 128.0f);
  RangeOpModel<int8_t> model({TensorType_INT8, {}, 0.0f, 128.0f},
                             {TensorType_INT8, {}, 0.0f, 128.0f},
                             {TensorType_INT8, {}, 0.0f, 128.0f});
  model.QuantizeAndPopulate<int8_t>(model.start(), {0.0f});
  model.QuantizeAndPopulate<int8_t>(model.limit(), {4.0f});
  model.QuantizeAndPopulate<int8_t>(model.delta(), {1.5f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(
      model.ExtractDequantVector<float>(model.output()),
      ElementsAreArray(ArrayFloatNear({0.0, 1.5, 3.0}, kQuantizedTolerance)));
}

TEST(RangeOpModel, SimpleInt16) {
  RangeOpModel<int16_t> model(TensorType_INT16);
  model.PopulateTensor<int16_t>(model.start(), {0});
  model.PopulateTensor<int16_t>(model.limit(), {4});
  model.PopulateTensor<int16_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, SimpleFloat16) {
  RangeOpModel<Eigen::half> model(TensorType_FLOAT16);
  model.PopulateTensor<Eigen::half>(model.start(), {Eigen::half(0)});
  model.PopulateTensor<Eigen::half>(model.limit(), {Eigen::half(4)});
  model.PopulateTensor<Eigen::half>(model.delta(), {Eigen::half(1)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(Eigen::half(0), Eigen::half(1),
                                             Eigen::half(2), Eigen::half(3)));
}

TEST(RangeOpModel, SimpleBFloat16) {
  RangeOpModel<Eigen::bfloat16> model(TensorType_BFLOAT16);
  model.PopulateTensor<Eigen::bfloat16>(model.start(), {Eigen::bfloat16(0)});
  model.PopulateTensor<Eigen::bfloat16>(model.limit(), {Eigen::bfloat16(4)});
  model.PopulateTensor<Eigen::bfloat16>(model.delta(), {Eigen::bfloat16(1)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(Eigen::bfloat16(0), Eigen::bfloat16(1),
                          Eigen::bfloat16(2), Eigen::bfloat16(3)));
}

TEST(RangeOpModel, Int8QuantizedDeltaGreaterThanOneConst) {
  RangeOpModel<int8_t> model({TensorType_INT8, {}, 0, 128},
                             {TensorType_INT8, {}, 0, 128},
                             {TensorType_INT8, {}, 0, 128});
  model.QuantizeAndPopulate<int8_t>(model.start(), {2});
  model.QuantizeAndPopulate<int8_t>(model.limit(), {9});
  model.QuantizeAndPopulate<int8_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.ExtractDequantVector<int8_t>(model.output()),
              ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int8DeltaGreaterThanOneConst) {
  RangeOpModel<int8_t> model(TensorType_INT8, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int16QuantizedDeltaGreaterThanOneConst) {
  RangeOpModel<int16_t> model({TensorType_INT16, {}, 0, 128},
                              {TensorType_INT16, {}, 0, 128},
                              {TensorType_INT16, {}, 0, 128});
  model.QuantizeAndPopulate<int16_t>(model.start(), {2});
  model.QuantizeAndPopulate<int16_t>(model.limit(), {9});
  model.QuantizeAndPopulate<int16_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.ExtractDequantVector<int16_t>(model.output()),
              ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int16DeltaGreaterThanOneConst) {
  RangeOpModel<int16_t> model(TensorType_INT16, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Float16DeltaGreaterThanOneConst) {
  RangeOpModel<Eigen::half> model(TensorType_FLOAT16, {Eigen::half(2)},
                                  {Eigen::half(9)}, {Eigen::half(2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(Eigen::half(2), Eigen::half(4),
                                             Eigen::half(6), Eigen::half(8)));
}

TEST(RangeOpModel, BFloat16DeltaGreaterThanOneConst) {
  RangeOpModel<Eigen::bfloat16> model(TensorType_BFLOAT16, {Eigen::bfloat16(2)},
                                      {Eigen::bfloat16(9)},
                                      {Eigen::bfloat16(2)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(Eigen::bfloat16(2), Eigen::bfloat16(4),
                          Eigen::bfloat16(6), Eigen::bfloat16(8)));
}

TEST(RangeOpModel, Int8EmptyOutputConstExample1) {
  RangeOpModel<int8_t> model({TensorType_INT8, {}, 0, 128},
                             {TensorType_INT8, {}, 0, 128},
                             {TensorType_INT8, {}, 0, 128});
  model.QuantizeAndPopulate<int8_t>(model.start(), {0});
  model.QuantizeAndPopulate<int8_t>(model.limit(), {0});
  model.QuantizeAndPopulate<int8_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.ExtractDequantVector<int8_t>(model.output()),
              ElementsAre());
}

TEST(RangeOpModel, Int8EmptyOutputConstExample2) {
  RangeOpModel<int8_t> model(TensorType_INT8, {0}, {0}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, Int16EmptyOutputConstExample1) {
  RangeOpModel<int16_t> model({TensorType_INT16, {}, 0, 128},
                              {TensorType_INT16, {}, 0, 128},
                              {TensorType_INT16, {}, 0, 128});
  model.QuantizeAndPopulate<int16_t>(model.start(), {0});
  model.QuantizeAndPopulate<int16_t>(model.limit(), {0});
  model.QuantizeAndPopulate<int16_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.ExtractDequantVector<int16_t>(model.output()),
              ElementsAre());
}

TEST(RangeOpModel, Int16EmptyOutputConstExample2) {
  RangeOpModel<int16_t> model(TensorType_INT16, {0}, {0}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, Float16EmptyOutputConst) {
  RangeOpModel<Eigen::half> model(TensorType_FLOAT16, {Eigen::half(0)},
                                  {Eigen::half(0)}, {Eigen::half(1)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, BFloat16EmptyOutputConst) {
  RangeOpModel<Eigen::bfloat16> model(TensorType_BFLOAT16, {Eigen::bfloat16(0)},
                                      {Eigen::bfloat16(0)},
                                      {Eigen::bfloat16(1)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

}  // namespace
}  // namespace tflite
