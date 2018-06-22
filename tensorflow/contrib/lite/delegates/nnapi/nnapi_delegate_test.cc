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
#include "tensorflow/contrib/lite/delegates/nnapi/nnapi_delegate.h"
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// TODO(b/110368244): figure out how to share the existing tests in kernels/ but
// with the delegation on. Also, add more unit tests to improve code coverage.

class FloatAddOpModel : public SingleOpModel {
 public:
  FloatAddOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_ADD, BuiltinOptions_AddOptions,
                 CreateAddOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

// Do a test with the NN API using no activation.
TEST(NNAPIDelegate, AddWithNoActivation) {
  FloatAddOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({-1.9, 0.4, 1.0, 1.3}));
}

// Do a test with the NN api with relu.
TEST(NNAPIDelegate, AddWithRelu) {
  FloatAddOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_RELU);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0.0, 0.4, 1.0, 1.3}));
}

class FloatMulOpModel : public SingleOpModel {
 public:
  FloatMulOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_MUL, BuiltinOptions_MulOptions,
                 CreateMulOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(NNAPIDelegate, MulWithNoActivation) {
  FloatMulOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-0.2, 0.04, 0.21, 0.4})));
}

class FloatPoolingOpModel : public SingleOpModel {
 public:
  FloatPoolingOpModel(BuiltinOperator type, const TensorData& input,
                      int filter_width, int filter_height,
                      const TensorData& output) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });

    input_ = AddInput(input);
    output_ = AddOutput(output);

    SetBuiltinOp(
        type, BuiltinOptions_Pool2DOptions,
        CreatePool2DOptions(builder_, Padding_VALID, 2, 2, filter_width,
                            filter_height, ActivationFunctionType_NONE)
            .Union());

    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int output_;
};

TEST(NNAPIDelegate, AveragePoolWithNoActivation) {
  FloatPoolingOpModel m(BuiltinOperator_AVERAGE_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2.75, 5.75}));
}

TEST(NNAPIDelegate, MaxPoolWithNoActivation) {
  FloatPoolingOpModel m(BuiltinOperator_MAX_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 10}));
}

TEST(NNAPIDelegate, L2PoolWithNoActivation) {
  FloatPoolingOpModel m(BuiltinOperator_L2_POOL_2D,
                        /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}},
                        /*filter_width=*/2, /*filter_height=*/2,
                        /*output=*/{TensorType_FLOAT32, {}});
  m.SetInput({
      0, 6, 2, 4,   //
      3, 2, 10, 7,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3.5, 6.5}));
}

class BaseConvolutionOpModel : public SingleOpModel {
 public:
  BaseConvolutionOpModel(
      const TensorData& input, const TensorData& filter,
      const TensorData& output, int stride_width = 2, int stride_height = 2,
      enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });

    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[0];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      auto bias_scale = GetScale(input_) * GetScale(filter_);
      TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);
    if (input.type != TensorType_FLOAT32) {
      // The following is required by quantized inference. It is the unittest's
      // responsibility to make sure the output scale falls into the correct
      // range.
      CHECK_LT(GetScale(input_) * GetScale(filter_), GetScale(output_));
    }

    SetBuiltinOp(BuiltinOperator_CONV_2D, BuiltinOptions_Conv2DOptions,
                 CreateConv2DOptions(
                     builder_, padding, stride_width, stride_height, activation,
                     dilation_width_factor, dilation_height_factor)
                     .Union());

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

class ConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetFilter(std::initializer_list<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

class QuantizedConvolutionOpModel : public BaseConvolutionOpModel {
 public:
  using BaseConvolutionOpModel::BaseConvolutionOpModel;

  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetFilter(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(filter_, data);
  }

  void SetBias(std::initializer_list<float> data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }

  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

// In this tests we set the input and output scales so that the results
// match exactly the 'non-quantized' version.
TEST(NNAPIDelegate, SimpleTestQuantized) {
  QuantizedConvolutionOpModel m({TensorType_UINT8, {2, 2, 4, 1}, -63.5, 64},
                                {TensorType_UINT8, {3, 2, 2, 1}, -63.5, 64},
                                {TensorType_UINT8, {}, -127, 128});
  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      18, 2, 5,  // first batch, left
                      18, 2, 5,  // first batch, right
                      17, 4, 3,  // second batch, left
                      37, 4, 3,  // second batch, right
                  },
                  1e-5)));
  // For good  measure, let's also verify the quantized values:
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 145, 129, 132,  //
                                 145, 129, 132,  //
                                 144, 131, 130,  //
                                 164, 131, 130,  //
                             }));
}

TEST(NNAPIDelegate, Conv2DWithNoActivation) {
  ConvolutionOpModel m({TensorType_FLOAT32, {2, 2, 4, 1}},
                       {TensorType_FLOAT32, {3, 2, 2, 1}},
                       {TensorType_FLOAT32, {}});

  m.SetInput({
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  });
  m.SetFilter({
      1, 2, 3, 4,    // first 2x2 filter
      -1, 1, -1, 1,  // second 2x2 filter
      -1, -1, 1, 1,  // third 2x2 filter
  });
  m.SetBias({1, 2, 3});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 18, 2, 5,  // first batch, left
                                 18, 2, 5,  // first batch, right
                                 17, 4, 3,  // second batch, left
                                 37, 4, 3,  // second batch, right
                             }));
}

class DepthwiseConvolutionOpModel : public SingleOpModel {
 public:
  DepthwiseConvolutionOpModel(const TensorData& input, const TensorData& filter,
                              const TensorData& output) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });

    input_ = AddInput(input);
    filter_ = AddInput(filter);

    int bias_size = GetShape(filter_)[3];
    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {bias_size}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      auto bias_scale = GetScale(input_) * GetScale(filter_);
      TensorData bias{TensorType_INT32, {bias_size}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    int input_depth = GetShape(input_)[3];
    int output_depth = GetShape(filter_)[3];
    int depth_mul = output_depth / input_depth;

    SetBuiltinOp(
        BuiltinOperator_DEPTHWISE_CONV_2D,
        BuiltinOptions_DepthwiseConv2DOptions,
        CreateDepthwiseConv2DOptions(builder_, Padding_VALID, 1, 1, depth_mul,
                                     ActivationFunctionType_NONE)
            .Union());

    BuildInterpreter({GetShape(input_), GetShape(filter_), GetShape(bias_)});
  }

  void SetFilter(std::initializer_list<float> f) { PopulateTensor(filter_, f); }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int filter_;
  int bias_;
  int output_;
};

TEST(NNAPIDelegate, DepthwiseConv2DWithNoActivation) {
  DepthwiseConvolutionOpModel m({TensorType_FLOAT32, {1, 3, 2, 2}},
                                {TensorType_FLOAT32, {1, 2, 2, 4}},
                                {TensorType_FLOAT32, {}});

  m.SetInput({
      1, 2, 7, 8,    // column 1
      3, 4, 9, 10,   // column 2
      5, 6, 11, 12,  // column 3
  });
  m.SetFilter({
      1, 2, 3, 4,        //
      -9, 10, -11, 12,   //
      5, 6, 7, 8,        //
      13, -14, 15, -16,  //
  });
  m.SetBias({1, 2, 3, 4});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 71, -34, 99, -20,  //
                                 91, -26, 127, -4,  //
                             }));
}

class FloatFullyConnectedOpModel : public SingleOpModel {
 public:
  FloatFullyConnectedOpModel(int units, int batches, const TensorData& input,
                             const TensorData& output = {TensorType_FLOAT32})
      : batches_(batches), units_(units) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });

    int total_input_size = 1;
    for (int i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ =
        AddInput({input.type, {units_, input_size_}, input.min, input.max});

    if (input.type == TensorType_FLOAT32) {
      bias_ = AddInput({TensorType_FLOAT32, {units_}});
    } else {
      // This is a quantized version. The scale of 'bias' depends on the scales
      // of input and filter. Supposedly this is correctly set during quantized
      // training.
      auto bias_scale = GetScale(input_) * GetScale(weights_);
      TensorData bias{TensorType_INT32, {units_}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, ActivationFunctionType_RELU)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(weights_), GetShape(bias_)});
  }

  int input_size() { return input_size_; }
  int num_units() { return units_; }
  int num_batches() { return batches_; }

  void SetBias(std::initializer_list<float> f) { PopulateTensor(bias_, f); }

  void SetWeights(std::initializer_list<float> f) {
    PopulateTensor(weights_, f);
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
};

TEST(NNAPIDelegate, FullyConnectedSimpleTest) {
  FloatFullyConnectedOpModel m(/*units=*/3, /*batches=*/2,
                               /*input=*/{TensorType_FLOAT32, {2, 10}});
  m.SetWeights({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
  });
  m.SetBias({1, 2, 3});

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAre(24, 25, 26, 58, 59, 60));
}

class SoftmaxOpModel : public SingleOpModel {
 public:
  SoftmaxOpModel(int batches, int size, float beta)
      : batches_(batches), input_size_(size), beta_(beta) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });

    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_SOFTMAX, BuiltinOptions_SoftmaxOptions,
                 CreateSoftmaxOptions(builder_, beta_).Union());
    BuildInterpreter({{batches_, input_size_}});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;

  int batches_;
  int input_size_;
  float beta_;
};

TEST(NNAPIDelegate, SoftmaxSimpleTest) {
  SoftmaxOpModel m(/*batches=*/2, /*size=*/5, /*beta=*/1.0);
  m.SetInput({
      1.0, 2.0, 3.0, 4.0, 5.0,       // b = 0
      -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 0
  });

  m.Invoke();

  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647,
           0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231},
          1e-6)));
}

class ReshapeOpModel : public SingleOpModel {
 public:
  ReshapeOpModel(std::initializer_list<int> input_shape,
                 std::initializer_list<int> new_shape) {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate());
    });

    input_ = AddInput(TensorType_FLOAT32);
    new_shape_ = AddInput(TensorType_INT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(builder_, builder_.CreateVector<int>(new_shape))
            .Union());
    BuildInterpreter({input_shape, {static_cast<int>(new_shape.size())}});
    PopulateTensor<int>(new_shape_, new_shape);
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int new_shape_;
  int output_;
};

TEST(NNAPIDelegate, ReshapeSimpleTest) {
  ReshapeOpModel m({1, 2, 4, 1}, {2, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
