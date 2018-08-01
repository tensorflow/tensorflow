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

class SingleOpModelWithNNAPI : public SingleOpModel {
 public:
  SingleOpModelWithNNAPI() {
    this->SetApplyDelegate([](Interpreter* interpreter) {
      interpreter->ModifyGraphWithDelegate(NnApiDelegate(), false);
    });
  }
};

class FloatAddOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatAddOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
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

class FloatMulOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatMulOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
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

class FloatPoolingOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatPoolingOpModel(BuiltinOperator type, const TensorData& input,
                      int filter_width, int filter_height,
                      const TensorData& output) {
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

class BaseConvolutionOpModel : public SingleOpModelWithNNAPI {
 public:
  BaseConvolutionOpModel(
      const TensorData& input, const TensorData& filter,
      const TensorData& output, int stride_width = 2, int stride_height = 2,
      enum Padding padding = Padding_VALID,
      enum ActivationFunctionType activation = ActivationFunctionType_NONE,
      int dilation_width_factor = 1, int dilation_height_factor = 1) {
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

class DepthwiseConvolutionOpModel : public SingleOpModelWithNNAPI {
 public:
  DepthwiseConvolutionOpModel(const TensorData& input, const TensorData& filter,
                              const TensorData& output) {
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

class FloatFullyConnectedOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatFullyConnectedOpModel(int units, int batches, const TensorData& input,
                             const TensorData& output = {TensorType_FLOAT32})
      : batches_(batches), units_(units) {
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

class SoftmaxOpModel : public SingleOpModelWithNNAPI {
 public:
  SoftmaxOpModel(int batches, int size, float beta)
      : batches_(batches), input_size_(size), beta_(beta) {
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

class ReshapeOpModel : public SingleOpModelWithNNAPI {
 public:
  ReshapeOpModel(std::initializer_list<int> input_shape,
                 std::initializer_list<int> new_shape) {
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

class SqueezeOpModel : public SingleOpModelWithNNAPI {
 public:
  SqueezeOpModel(const TensorData& input, const TensorData& output,
                 std::initializer_list<int> axis) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_SQUEEZE, BuiltinOptions_SqueezeOptions,
        CreateSqueezeOptions(builder_, builder_.CreateVector<int>(axis))
            .Union());
    BuildInterpreter({GetShape(input_)});
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

TEST(NNAPIDelegate, SqueezeSimpleTest) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SqueezeOpModel m({TensorType_FLOAT32, {1, 24, 1}}, {TensorType_FLOAT32, {24}},
                   {});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                        9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}));
}

TEST(NNAPIDelegate, SqueezeWithAxisTest) {
  std::initializer_list<float> data = {
      1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  SqueezeOpModel m({TensorType_FLOAT32, {1, 24, 1}}, {TensorType_FLOAT32, {24}},
                   {2});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                        9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0}));
}

class L2NormOpModel : public SingleOpModelWithNNAPI {
 public:
  L2NormOpModel(const TensorData& input, const TensorData& output,
                ActivationFunctionType activation_type) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_L2_NORMALIZATION, BuiltinOptions_L2NormOptions,
                 CreateL2NormOptions(builder_, activation_type).Union());
    BuildInterpreter({GetShape(input_)});
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

TEST(NNAPIDelegate, L2NormSimpleTest) {
  std::initializer_list<float> data = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
  L2NormOpModel m({TensorType_FLOAT32, {1, 1, 1, 6}},
                  {TensorType_FLOAT32, {1, 1, 1, 6}},
                  ActivationFunctionType_NONE);
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 6}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05}));
}

class TransposeSimpleModel : public SingleOpModelWithNNAPI {
 public:
  TransposeSimpleModel(std::initializer_list<int> input_shape,
                       std::initializer_list<int> perm_shape,
                       std::initializer_list<int> perm) {
    input_ = AddInput(TensorType_FLOAT32);
    perm_ = AddConstInput(TensorType_INT32, perm, perm_shape);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union());
    BuildInterpreter({input_shape, perm_shape});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int perm_;
  int output_;
};

TEST(NNAPIDelegate, TransposeSimpleTest) {
  TransposeSimpleModel m({2, 3, 4}, {3}, {2, 0, 1});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

class FloatSubOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatSubOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_SUB, BuiltinOptions_SubOptions,
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

TEST(NNAPIDelegate, SubWithNoActivation) {
  FloatSubOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.7, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.3, 0.5});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({-2.1, 0.0, 0.4, 0.3})));
}

class FloatDivOpModel : public SingleOpModelWithNNAPI {
 public:
  FloatDivOpModel(const TensorData& input1, const TensorData& input2,
                  const TensorData& output,
                  ActivationFunctionType activation_type) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_DIV, BuiltinOptions_DivOptions,
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

TEST(NNAPIDelegate, DivWithNoActivation) {
  FloatDivOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {1, 2, 2, 1}},
                    {TensorType_FLOAT32, {}}, ActivationFunctionType_NONE);
  m.PopulateTensor<float>(m.input1(), {-2.0, 0.2, 0.8, 0.8});
  m.PopulateTensor<float>(m.input2(), {0.1, 0.2, 0.4, 0.2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({-20, 1, 2, 4})));
}

class BaseConcatenationOpModel : public SingleOpModelWithNNAPI {
 public:
  BaseConcatenationOpModel() {}
  BaseConcatenationOpModel(const TensorData& input_template, int axis,
                           int num_inputs) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < num_inputs; ++i) {
      all_input_shapes.push_back(input_template.shape);
      AddInput(input_template);
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }

 protected:
  int output_;
};

class ConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  void SetInput(int index, std::initializer_list<float> data) {
    PopulateTensor(index, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

TEST(NNAPIDelegate, ConcatenationThreeDimensionalOneInput) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/1,
                          /*num_inputs=*/1);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({1, 3, 4, 7}));
}

TEST(NNAPIDelegate, ConcatenationFourInputs) {
  ConcatenationOpModel m0({TensorType_FLOAT32, {2, 1, 2}}, /*axis=*/2,
                          /*num_inputs=*/4);
  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetOutput(),
              ElementsAreArray({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              }));
}

class QuantizedConcatenationOpModel : public BaseConcatenationOpModel {
 public:
  using BaseConcatenationOpModel::BaseConcatenationOpModel;
  QuantizedConcatenationOpModel(const std::vector<TensorData>& input_template,
                                int axis, int num_inputs,
                                const TensorData& output_template) {
    std::vector<std::vector<int>> all_input_shapes;
    CHECK_EQ(input_template.size(), num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      all_input_shapes.push_back(input_template[i].shape);
      AddInput(input_template[i]);
    }
    output_ = AddOutput({output_template.type, /*shape=*/{},
                         output_template.min, output_template.max});
    SetBuiltinOp(
        BuiltinOperator_CONCATENATION, BuiltinOptions_ConcatenationOptions,
        CreateConcatenationOptions(builder_, axis, ActivationFunctionType_NONE)
            .Union());
    BuildInterpreter(all_input_shapes);
  }
  void SetInput(int index, std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(index, data);
  }
  std::vector<uint8_t> GetOutput() { return ExtractVector<uint8_t>(output_); }
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }
};

TEST(NNAPIDelegate, ConcatenationFourInputsQuantized) {
  QuantizedConcatenationOpModel m0({TensorType_UINT8, {2, 1, 2}, -12.7, 12.8},
                                   /*axis=*/2,
                                   /*num_inputs=*/4);

  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({
                                  137, 157, 138, 158, 139, 159, 140, 160,  //
                                  167, 197, 168, 198, 169, 199, 170, 200,  //
                              }));
}

TEST(NNAPIDelegate, ConcatenationFourInputsQuantizedMixedRange) {
  QuantizedConcatenationOpModel m0({{TensorType_UINT8, {2, 1, 2}, -10.7, 10.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 12.8},
                                    {TensorType_UINT8, {2, 1, 2}, -11, 11.8},
                                    {TensorType_UINT8, {2, 1, 2}, 0, 7.4}},
                                   /*axis=*/2, /*num_inputs=*/4,
                                   {TensorType_UINT8, {2, 1, 2}, -12.7, 12.8});

  m0.SetInput(0, {1.0f, 3.0f, 4.0f, 7.0f});
  m0.SetInput(1, {1.1f, 3.1f, 4.1f, 7.1f});
  m0.SetInput(2, {1.2f, 3.2f, 4.2f, 7.2f});
  m0.SetInput(3, {1.3f, 3.3f, 4.3f, 7.3f});
  m0.Invoke();
  EXPECT_THAT(m0.GetDequantizedOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.0f, 3.0f, 1.1f, 3.1f, 1.2f, 3.2f, 1.3f, 3.3f,  //
                  4.0f, 7.0f, 4.1f, 7.1f, 4.2f, 7.2f, 4.3f, 7.3f,  //
              })));
  EXPECT_THAT(m0.GetOutput(), ElementsAreArray({
                                  137, 157, 138, 158, 139, 159, 140, 160,  //
                                  167, 197, 168, 198, 169, 199, 170, 200,  //
                              }));
}

class DequantizeOpModel : public SingleOpModelWithNNAPI {
 public:
  DequantizeOpModel(std::initializer_list<int> shape, float min, float max) {
    input_ = AddInput({TensorType_UINT8, shape, min, max});
    output_ = AddOutput({TensorType_FLOAT32, shape});
    SetBuiltinOp(BuiltinOperator_DEQUANTIZE, BuiltinOptions_DequantizeOptions,
                 CreateDequantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<uint8_t> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(NNAPIDelegate, DequantizeFourDimensional) {
  DequantizeOpModel m({2, 5}, -63.5, 64);

  m.SetInput({0, 1, 2, 3, 4, 251, 252, 253, 254, 255});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64})));
}

class FloorOpModel : public SingleOpModelWithNNAPI {
 public:
  FloorOpModel(std::initializer_list<int> input_shape, TensorType input_type) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_FLOOR, BuiltinOptions_NONE, 0);
    BuildInterpreter({
        input_shape,
    });
  }

  int input() { return input_; }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(NNAPIDelegate, FloorSingleDim) {
  FloorOpModel model({2}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {8.5, 0.0});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({8, 0}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2}));
}

TEST(NNAPIDelegate, FloorMultiDims) {
  FloorOpModel model({2, 1, 1, 5}, TensorType_FLOAT32);
  model.PopulateTensor<float>(model.input(), {
                                                 0.0001,
                                                 8.0001,
                                                 0.9999,
                                                 9.9999,
                                                 0.5,
                                                 -0.0001,
                                                 -8.0001,
                                                 -0.9999,
                                                 -9.9999,
                                                 -0.5,
                                             });
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({0, 8, 0, 9, 0, -1, -9, -1, -10, -1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 1, 5}));
}

class LocalResponseNormOpModel : public SingleOpModelWithNNAPI {
 public:
  LocalResponseNormOpModel(std::initializer_list<int> input_shape, int radius,
                           float bias, float alpha, float beta) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
                 BuiltinOptions_LocalResponseNormalizationOptions,
                 CreateLocalResponseNormalizationOptions(builder_, radius, bias,
                                                         alpha, beta)
                     .Union());
    BuildInterpreter({input_shape});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(NNAPIDelegate, LocalResponseNormSameAsL2Norm) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/0.0,
                             /*alpha=*/1.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  // The result is every input divided by 2.
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05})));
}

TEST(NNAPIDelegate, LocalResponseNormWithAlpha) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/0.0,
                             /*alpha=*/4.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  // The result is every input divided by 3.
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {-0.275, 0.15, 0.175, 0.3, -0.175, 0.025})));
}

TEST(NNAPIDelegate, LocalResponseNormWithBias) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/20, /*bias=*/9.0,
                             /*alpha=*/4.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  // The result is every input divided by 5.
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({-0.22, 0.12, 0.14, 0.24, -0.14, 0.02})));
}

TEST(NNAPIDelegate, LocalResponseNormSmallRadius) {
  LocalResponseNormOpModel m({1, 1, 1, 6}, /*radius=*/2, /*bias=*/9.0,
                             /*alpha=*/4.0, /*beta=*/0.5);
  m.SetInput({-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear(
          {-0.264926, 0.125109, 0.140112, 0.267261, -0.161788, 0.0244266})));
}

class LSHProjectionOpModel : public SingleOpModelWithNNAPI {
 public:
  LSHProjectionOpModel(LSHProjectionType type,
                       std::initializer_list<int> hash_shape,
                       std::initializer_list<int> input_shape,
                       std::initializer_list<int> weight_shape) {
    hash_ = AddInput(TensorType_FLOAT32);
    input_ = AddInput(TensorType_INT32);
    if (weight_shape.size() > 0) {
      weight_ = AddInput(TensorType_FLOAT32);
    }
    output_ = AddOutput(TensorType_INT32);

    SetBuiltinOp(BuiltinOperator_LSH_PROJECTION,
                 BuiltinOptions_LSHProjectionOptions,
                 CreateLSHProjectionOptions(builder_, type).Union());
    if (weight_shape.size() > 0) {
      BuildInterpreter({hash_shape, input_shape, weight_shape});
    } else {
      BuildInterpreter({hash_shape, input_shape});
    }

    output_size_ = 1;
    for (int i : hash_shape) {
      output_size_ *= i;
      if (type == LSHProjectionType_SPARSE) {
        break;
      }
    }
  }
  void SetInput(std::initializer_list<int> data) {
    PopulateTensor(input_, data);
  }

  void SetHash(std::initializer_list<float> data) {
    PopulateTensor(hash_, data);
  }

  void SetWeight(std::initializer_list<float> f) { PopulateTensor(weight_, f); }

  std::vector<int> GetOutput() { return ExtractVector<int>(output_); }

 private:
  int input_;
  int hash_;
  int weight_;
  int output_;

  int output_size_;
};

TEST(NNAPIDelegate, LSHProjectionDense1DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_DENSE, {3, 2}, {5}, {5});

  m.SetInput({12345, 54321, 67890, 9876, -12345678});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});
  m.SetWeight({1.0, 1.0, 1.0, 1.0, 1.0});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0, 0, 1, 0, 0));
}

TEST(NNAPIDelegate, LSHProjectionSparse1DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_SPARSE, {3, 2}, {5}, {});

  m.SetInput({12345, 54321, 67890, 9876, -12345678});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 0, 4 + 1, 8 + 0));
}

TEST(NNAPIDelegate, LSHProjectionSparse3DInputs) {
  LSHProjectionOpModel m(LSHProjectionType_SPARSE, {3, 2}, {5, 2, 2}, {5});

  m.SetInput({1234, 2345, 3456, 1234, 4567, 5678, 6789, 4567, 7891, 8912,
              9123, 7890, -987, -876, -765, -987, -543, -432, -321, -543});
  m.SetHash({0.123, 0.456, -0.321, 1.234, 5.678, -4.321});
  m.SetWeight({0.12, 0.34, 0.56, 0.67, 0.78});

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAre(0 + 2, 4 + 1, 8 + 1));
}

class BaseActivationsOpModel : public SingleOpModelWithNNAPI {
 public:
  // Most activations don't take any options, so this constructor works for
  // them.
  BaseActivationsOpModel(BuiltinOperator type, TensorData input) {
    input_ = AddInput(input);
    if (input.type == TensorType_UINT8) {
      output_ = AddOutput({input.type, {}, 0, 0, 1. / 256});
    } else {
      output_ = AddOutput({input.type, {}});
    }
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

  BaseActivationsOpModel(BuiltinOperator type, const TensorData& input,
                         const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input_)});
  }

 protected:
  int input_;
  int output_;
};

class FloatActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
};

const float kQuantizedTolerance = 2 * (1. / 256);

class QuantizedActivationsOpModel : public BaseActivationsOpModel {
 public:
  using BaseActivationsOpModel::BaseActivationsOpModel;

  template <typename T>
  void SetInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }
  template <typename T>

  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }
  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }
};

TEST(NNAPIDelegate, Relu) {
  FloatActivationsOpModel m(BuiltinOperator_RELU,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,   //
                                 3, 0, 10, 1,  //
                             }));
}

TEST(NNAPIDelegate, Relu1) {
  FloatActivationsOpModel m(BuiltinOperator_RELU_N1_TO_1,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0.0, -0.6, 0.2, -0.4,  //
      0.3, -2.0, 1.1, -0.1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0.0, -0.6, 0.2, -0.4,  //
                                 0.3, -1.0, 1.0, -0.1,  //
                             }));
}

TEST(NNAPIDelegate, Relu6) {
  FloatActivationsOpModel m(BuiltinOperator_RELU6,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 2, 4,  //
                                 3, 0, 6, 1,  //
                             }));
}

TEST(NNAPIDelegate, Tanh) {
  FloatActivationsOpModel m(BuiltinOperator_TANH,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0, -0.9999877, 0.9640275, 0.999329,    //
                                 0.99505475, -0.9640275, 1, 0.7615941,  //
                             })));
}

TEST(NNAPIDelegate, LogisticFloat) {
  FloatActivationsOpModel m(BuiltinOperator_LOGISTIC,
                            /*input=*/{TensorType_FLOAT32, {1, 2, 4, 1}});
  m.SetInput({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.5, 0.002473, 0.880797, 0.982014,       //
                                 0.952574, 0.119203, 0.999955, 0.731059,  //
                             })));
}

TEST(NNAPIDelegate, LogisticQuantized) {
  QuantizedActivationsOpModel m(
      BuiltinOperator_LOGISTIC,
      /*input=*/{TensorType_UINT8, {1, 2, 4, 1}, -10, 10});
  m.SetInput<uint8_t>({
      0, -6, 2, 4,   //
      3, -2, 10, 1,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      0.5, 0.002473, 0.880797, 0.982014,       //
                      0.952574, 0.119203, 0.999955, 0.731059,  //
                  },
                  kQuantizedTolerance)));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({128, 1, 227, 251, 244, 32, 255, 188}));
}

#if 0
class ResizeBilinearOpModel : public SingleOpModelWithNNAPI {
 public:
  ResizeBilinearOpModel(const TensorData& input,
                        std::initializer_list<int> size_data = {}) {
    bool const_size = size_data.size() != 0;
    input_ = AddInput(input);
    if (const_size) {
      size_ = AddConstInput(TensorType_INT32, size_data, {2});
    } else {
      size_ = AddInput({TensorType_INT32, {2}});
    }
    output_ = AddOutput(input.type);
    SetBuiltinOp(BuiltinOperator_RESIZE_BILINEAR,
                 BuiltinOptions_ResizeBilinearOptions,
                 CreateResizeBilinearOptions(builder_).Union());
    if (const_size) {
      BuildInterpreter({GetShape(input_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(size_)});
    }
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }
  void SetSize(std::initializer_list<int> data) { PopulateTensor(size_, data); }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int size_;
  int output_;
};

TEST(NNAPIDelegate, ResizeBilinearHorizontal) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 1, 2, 1}});
  m.SetInput<float>({3, 6});
  m.SetSize({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6})));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 1, 2, 1}}, {1, 3});
  const_m.SetInput<float>({3, 6});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 5, 6})));
}

TEST(NNAPIDelegate, ResizeBilinearVertical) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 1, 1}});
  m.SetInput<float>({3, 9});
  m.SetSize({3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9})));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 2, 1, 1}}, {3, 1});
  const_m.SetInput<float>({3, 9});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 7, 9})));
}

TEST(NNAPIDelegate, ResizeBilinearTwoDimensional) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}});
  m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                        3, 5, 6,    //
                                        7, 9, 10,   //
                                        9, 11, 12,  //
                                    })));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 2, 2, 1}}, {3, 3});
  const_m.SetInput<float>({
      3, 6,  //
      9, 12  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({
                                              3, 5, 6,    //
                                              7, 9, 10,   //
                                              9, 11, 12,  //
                                          })));
}
#endif

template <typename T>
class PadOpModel : public SingleOpModelWithNNAPI {
 public:
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetQuantizedPadValue(float data) {
    QuantizeAndPopulate<uint8_t>(constant_values_, {data});
  }

  void SetPaddings(std::initializer_list<int> paddings) {
    PopulateTensor<int>(paddings_, paddings);
  }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
  int paddings_;
  int constant_values_;
};

class PadOpConstModel : public PadOpModel<float> {
 public:
  PadOpConstModel(const TensorData& input,
                  std::initializer_list<int> paddings_shape,
                  std::initializer_list<int> paddings,
                  const TensorData& output) {
    input_ = AddInput(input);
    paddings_ = AddConstInput(TensorType_INT32, paddings, paddings_shape);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                 CreatePadOptions(builder_).Union());
    BuildInterpreter({input.shape});
  }
};

TEST(NNAPIDelegate, PadAdvancedConstTest) {
  PadOpConstModel m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2},
                    {0, 0, 0, 2, 1, 3, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

class SpaceToBatchNDOpModel : public SingleOpModelWithNNAPI {
 public:
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetBlockShape(std::initializer_list<int> data) {
    PopulateTensor<int>(block_shape_, data);
  }

  void SetPaddings(std::initializer_list<int> data) {
    PopulateTensor<int>(paddings_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int block_shape_;
  int paddings_;
  int output_;
};

class SpaceToBatchNDOpConstModel : public SpaceToBatchNDOpModel {
 public:
  SpaceToBatchNDOpConstModel(std::initializer_list<int> input_shape,
                             std::initializer_list<int> block_shape,
                             std::initializer_list<int> paddings) {
    input_ = AddInput(TensorType_FLOAT32);
    block_shape_ = AddConstInput(TensorType_INT32, block_shape, {2});
    paddings_ = AddConstInput(TensorType_INT32, paddings, {2, 2});
    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_SPACE_TO_BATCH_ND,
                 BuiltinOptions_SpaceToBatchNDOptions,
                 CreateSpaceToBatchNDOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }
};

TEST(NNAPIDelegate, SpaceToBatchNDSimpleConstTest) {
  SpaceToBatchNDOpConstModel m({1, 4, 4, 1}, {2, 2}, {0, 0, 0, 0});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(NNAPIDelegate, SpaceToBatchNDMultipleInputBatchesConstTest) {
  SpaceToBatchNDOpConstModel m({2, 2, 4, 1}, {2, 2}, {0, 0, 0, 0});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(NNAPIDelegate, SpaceToBatchNDSimplePaddingConstTest) {
  SpaceToBatchNDOpConstModel m({1, 5, 2, 1}, {3, 2}, {1, 0, 2, 0});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
                                 0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10,
                             }));
}

TEST(NNAPIDelegate, SpaceToBatchNDComplexPaddingConstTest) {
  SpaceToBatchNDOpConstModel m({1, 4, 2, 1}, {3, 2}, {1, 1, 2, 4});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
                                 0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
                                 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
                             }));
}

template <typename input_type = float,
          TensorType tensor_input_type = TensorType_FLOAT32>
class StridedSliceOpModel : public SingleOpModelWithNNAPI {
 public:
  StridedSliceOpModel(std::initializer_list<int> input_shape,
                      std::initializer_list<int> begin_shape,
                      std::initializer_list<int> end_shape,
                      std::initializer_list<int> strides_shape, int begin_mask,
                      int end_mask, int ellipsis_mask, int new_axis_mask,
                      int shrink_axis_mask) {
    input_ = AddInput(tensor_input_type);
    begin_ = AddInput(TensorType_INT32);
    end_ = AddInput(TensorType_INT32);
    strides_ = AddInput(TensorType_INT32);
    output_ = AddOutput(tensor_input_type);
    SetBuiltinOp(
        BuiltinOperator_STRIDED_SLICE, BuiltinOptions_StridedSliceOptions,
        CreateStridedSliceOptions(builder_, begin_mask, end_mask, ellipsis_mask,
                                  new_axis_mask, shrink_axis_mask)
            .Union());
    BuildInterpreter({input_shape, begin_shape, end_shape, strides_shape});
  }

  void SetInput(std::initializer_list<input_type> data) {
    PopulateTensor<input_type>(input_, data);
  }
  void SetBegin(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(begin_, data);
  }
  void SetEnd(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(end_, data);
  }
  void SetStrides(std::initializer_list<int32_t> data) {
    PopulateTensor<int32_t>(strides_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int begin_;
  int end_;
  int strides_;
  int output_;
};

TEST(NNAPIDelegate, StridedSliceIn2D) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 0);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({1, 0});
  m.SetEnd({2, 2});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
}

TEST(NNAPIDelegate, StridedSliceIn2D_ShrinkAxis_NegativeSlice) {
  // This is equivalent to tf.range(4)[:, tf.newaxis][-2, -1].
  StridedSliceOpModel<> m({4, 1}, {2}, {2}, {2}, 0, 0, 0, 0, 3);
  m.SetInput({0, 1, 2, 3});
  m.SetBegin({-2, -1});
  m.SetEnd({-1, 0});
  m.SetStrides({1, 1});

  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2}));
}

TEST(NNAPIDelegate, StridedSliceIn2D_ShrinkAxisMask) {
  StridedSliceOpModel<> m({2, 3}, {2}, {2}, {2}, 0, 0, 0, 0, 3);
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetBegin({0, 0});
  m.SetEnd({1, 1});
  m.SetStrides({1, 1});
  m.Invoke();
  EXPECT_TRUE(m.GetOutputShape().empty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
