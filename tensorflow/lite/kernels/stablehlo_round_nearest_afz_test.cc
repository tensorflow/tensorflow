#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class RoundNearestAfzOpModel : public SingleOpModel {
 public:
  RoundNearestAfzOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_ROUND_NEAREST_AFZ,
                 BuiltinOptions_NONE, 0);
    SetBypassDefaultDelegates();
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  template <typename Datatype>
  std::vector<Datatype> GetOutput() {
    return ExtractVector<Datatype>(output_);
  }

 protected:
  int input_;
  int output_;
};

TEST(StablehloRoundTest, RoundFloat32) {
  RoundNearestAfzOpModel model({TensorType_FLOAT32, {1, 5}},
                               {TensorType_FLOAT32, {}});
  model.PopulateTensor<float>(model.input(), {-2.5f, 0.4f, 0.5f, 0.6f, 2.5f});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutput<float>(),
              ElementsAre(-3.0f, 0.0f, 1.0f, 1.0f, 3.0f));
}

TEST(StablehloRoundTest, RoundFloat16) {
  RoundNearestAfzOpModel model({TensorType_FLOAT16, {1, 5}},
                               {TensorType_FLOAT16, {}});
  model.PopulateTensor<Eigen::half>(
      model.input(), {Eigen::half(-3.644530e+00), Eigen::half(-1.992190e+00),
                      Eigen::half(-1.826170e+00), Eigen::half(3.039060e+00),
                      Eigen::half(1.963870e+00)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput<Eigen::half>(),
      ElementsAre(Eigen::half(-4.000000e+00), Eigen::half(-2.000000e+00),
                  Eigen::half(-2.000000e+00), Eigen::half(3.000000e+00),
                  Eigen::half(2.000000e+00)));
}

TEST(StablehloRoundTest, RoundBFloat16) {
  RoundNearestAfzOpModel model({TensorType_BFLOAT16, {1, 5}},
                               {TensorType_BFLOAT16, {}});
  model.PopulateTensor<Eigen::bfloat16>(
      model.input(),
      {Eigen::bfloat16(3.500000e+00), Eigen::bfloat16(-6.437500e+00),
       Eigen::bfloat16(9.453120e-01), Eigen::bfloat16(-1.367190e+00),
       Eigen::bfloat16(1.218750e+00)});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      model.GetOutput<Eigen::bfloat16>(),
      ElementsAre(Eigen::bfloat16(4.000000e+00), Eigen::bfloat16(-6.000000e+00),
                  Eigen::bfloat16(1.000000e+00), Eigen::bfloat16(-1.000000e+00),
                  Eigen::bfloat16(1.000000e+00)));
}

class QuantizedRoundOpModel : public RoundNearestAfzOpModel {
 public:
  QuantizedRoundOpModel(TensorData input, TensorData output)
      : RoundNearestAfzOpModel(SymmetricInt16Scaling(std::move(input)),
                        SymmetricInt16Scaling(std::move(output))) {}

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

 private:
  TensorData SymmetricInt16Scaling(TensorData tensor) {
    if (tensor.type == TensorType_INT16) {
      CHECK_EQ(std::abs(tensor.min), tensor.max);
      tensor.scale = tensor.max / std::numeric_limits<int16_t>::max();
      tensor.zero_point = 0;
      tensor.min = 0;
      tensor.max = 0;
    }
    return tensor;
  }
};

template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <TensorType tensor_type, typename integer_dtype>
void QuantizedTestsNoActivation() {
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-7.0, 7.0);
  std::vector<float> input = {-6.5, -5.5, 0.5, 1.2, 4.0, 4.5};
  std::vector<float> result = {-6.0, -5.0, 0, 1.0, 4.0, 6.0};

  QuantizedRoundOpModel model({tensor_type, {6}, -7.0, 7.0},
                             {tensor_type, {}, -7.0, 7.0});
  model.QuantizeAndPopulate<integer_dtype>(model.input(), input);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetDequantizedOutput<integer_dtype>(),
              ElementsAreArray(ArrayFloatNear(result, kQuantizedTolerance)));
}

TEST(QuantizedRoundOpModel, QuantizedTestsNoActivationInt8) {
  QuantizedTestsNoActivation<TensorType_INT8, int8_t>();
}

TEST(QuantizedRoundOpModel, QuantizedTestsNoActivationInt16) {
  QuantizedTestsNoActivation<TensorType_INT16, int16_t>();
}

}  // namespace
}  // namespace tflite
