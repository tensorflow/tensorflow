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
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

namespace {

class DepthwiseConv2DTester {
 public:
  DepthwiseConv2DTester() = default;
  DepthwiseConv2DTester(const DepthwiseConv2DTester&) = delete;
  DepthwiseConv2DTester& operator=(const DepthwiseConv2DTester&) = delete;

  DepthwiseConv2DTester& BatchSize(int32_t batch_size) {
    EXPECT_GT(batch_size, 0);
    batch_size_ = batch_size;
    return *this;
  }

  int32_t BatchSize() const { return batch_size_; }

  DepthwiseConv2DTester& Groups(int32_t groups) {
    EXPECT_GT(groups, 0);
    groups_ = groups;
    return *this;
  }

  int32_t Groups() const { return groups_; }

  DepthwiseConv2DTester& DepthMultiplier(int32_t depth_multiplier) {
    EXPECT_GT(depth_multiplier, 0);
    depth_multiplier_ = depth_multiplier;
    return *this;
  }

  int32_t DepthMultiplier() const { return depth_multiplier_; }

  int32_t InputChannels() const { return Groups(); }

  int32_t OutputChannels() const { return DepthMultiplier() * Groups(); }

  DepthwiseConv2DTester& InputHeight(int32_t input_height) {
    EXPECT_GT(input_height, 0);
    input_height_ = input_height;
    return *this;
  }

  int32_t InputHeight() const { return input_height_; }

  DepthwiseConv2DTester& InputWidth(int32_t input_width) {
    EXPECT_GT(input_width, 0);
    input_width_ = input_width;
    return *this;
  }

  int32_t InputWidth() const { return input_width_; }

  int32_t OutputWidth() const {
    const int32_t output_width = (InputWidth() - 1) / StrideWidth() + 1;
    EXPECT_GT(output_width, 0);
    return output_width;
  }

  int32_t OutputHeight() const {
    const int32_t output_height = (InputHeight() - 1) / StrideHeight() + 1;
    EXPECT_GT(output_height, 0);
    return output_height;
  }

  DepthwiseConv2DTester& KernelHeight(int32_t kernel_height) {
    EXPECT_GT(kernel_height, 0);
    kernel_height_ = kernel_height;
    return *this;
  }

  int32_t KernelHeight() const { return kernel_height_; }

  DepthwiseConv2DTester& KernelWidth(int32_t kernel_width) {
    EXPECT_GT(kernel_width, 0);
    kernel_width_ = kernel_width;
    return *this;
  }

  int32_t KernelWidth() const { return kernel_width_; }

  DepthwiseConv2DTester& StrideHeight(int32_t stride_height) {
    EXPECT_GT(stride_height, 0);
    stride_height_ = stride_height;
    return *this;
  }

  int32_t StrideHeight() const { return stride_height_; }

  DepthwiseConv2DTester& StrideWidth(int32_t stride_width) {
    EXPECT_GT(stride_width, 0);
    stride_width_ = stride_width;
    return *this;
  }

  int32_t StrideWidth() const { return stride_width_; }

  DepthwiseConv2DTester& DilationHeight(int32_t dilation_height) {
    EXPECT_GT(dilation_height, 0);
    dilation_height_ = dilation_height;
    return *this;
  }

  int32_t DilationHeight() const { return dilation_height_; }

  DepthwiseConv2DTester& DilationWidth(int32_t dilation_width) {
    EXPECT_GT(dilation_width, 0);
    dilation_width_ = dilation_width;
    return *this;
  }

  int32_t DilationWidth() const { return dilation_width_; }

  void Test(TfLiteDelegate* delegate) const {
    ASSERT_EQ(DepthMultiplier(), 1) << "Flow does not support depth multiplier";

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<char> buffer = CreateTfLiteModel(std::ref(f32rng));
    const Model* model = GetModel(buffer.data());

    std::unique_ptr<Interpreter> delegate_interpreter;
    ASSERT_EQ(
        InterpreterBuilder(model, ::tflite::ops::builtin::BuiltinOpResolver())(
            &delegate_interpreter),
        kTfLiteOk);
    std::unique_ptr<Interpreter> default_interpreter;
    ASSERT_EQ(
        InterpreterBuilder(model, ::tflite::ops::builtin::BuiltinOpResolver())(
            &default_interpreter),
        kTfLiteOk);

    ASSERT_TRUE(delegate_interpreter);
    ASSERT_TRUE(default_interpreter);

    ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
    ASSERT_EQ(default_interpreter->inputs().size(), 1);

    ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
    ASSERT_EQ(default_interpreter->outputs().size(), 1);

    ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

    ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate),
              kTfLiteOk);

    float* default_input_data = default_interpreter->typed_tensor<float>(
        default_interpreter->inputs()[0]);
    std::generate(default_input_data,
                  default_input_data + BatchSize() * InputChannels() *
                                           InputHeight() * InputWidth(),
                  std::ref(f32rng));

    float* xnnpack_input_data = delegate_interpreter->typed_tensor<float>(
        delegate_interpreter->inputs()[0]);
    std::copy(default_input_data,
              default_input_data +
                  BatchSize() * InputChannels() * InputHeight() * InputWidth(),
              xnnpack_input_data);

    default_interpreter->Invoke();
    delegate_interpreter->Invoke();

    float* default_output_data = default_interpreter->typed_tensor<float>(
        default_interpreter->outputs()[0]);
    float* xnnpack_output_data = delegate_interpreter->typed_tensor<float>(
        delegate_interpreter->outputs()[0]);

    for (size_t i = 0;
         i < BatchSize() * OutputChannels() * OutputHeight() * OutputWidth();
         i++) {
      ASSERT_NEAR(default_output_data[i], xnnpack_output_data[i],
                  std::numeric_limits<float>::epsilon() *
                      std::max(std::abs(default_output_data[i]) * 10.0f, 1.0f));
    }
  }

 private:
  std::vector<char> CreateTfLiteModel(std::function<float()> f32rng) const {
    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Offset<OperatorCode> operator_code =
        CreateOperatorCode(builder, BuiltinOperator_DEPTHWISE_CONV_2D, 0);

    flatbuffers::Offset<DepthwiseConv2DOptions> depthwise_conv2d_options =
        CreateDepthwiseConv2DOptions(builder, Padding_SAME, StrideWidth(),
                                     StrideHeight(), DepthMultiplier(),
                                     ActivationFunctionType_NONE,
                                     DilationWidth(), DilationHeight());

    std::vector<float> filter_data(KernelHeight() * KernelWidth() *
                                   OutputChannels());
    std::vector<float> bias_data(OutputChannels());

    std::generate(filter_data.begin(), filter_data.end(), f32rng);
    std::generate(bias_data.begin(), bias_data.end(), f32rng);

    flatbuffers::Offset<Buffer> buffers[3] = {
        CreateBuffer(builder, builder.CreateVector({})),
        CreateBuffer(builder,
                     builder.CreateVector(
                         reinterpret_cast<const uint8_t*>(filter_data.data()),
                         sizeof(float) * filter_data.size())),
        CreateBuffer(builder,
                     builder.CreateVector(
                         reinterpret_cast<const uint8_t*>(bias_data.data()),
                         sizeof(float) * bias_data.size())),
    };

    const int32_t input_shape[4] = {BatchSize(), InputHeight(), InputWidth(),
                                    InputChannels()};
    const int32_t output_shape[4] = {BatchSize(), OutputHeight(), OutputWidth(),
                                     OutputChannels()};
    const int32_t filter_shape[4] = {1, KernelHeight(), KernelWidth(),
                                     OutputChannels()};
    const int32_t bias_shape[1] = {OutputChannels()};

    flatbuffers::Offset<Tensor> tensors[4] = {
        CreateTensor(builder, builder.CreateVector<int32_t>(input_shape, 4),
                     TensorType_FLOAT32, /*buffer=*/0,
                     builder.CreateString("X")),
        CreateTensor(builder, builder.CreateVector<int32_t>(filter_shape, 4),
                     TensorType_FLOAT32, /*buffer=*/1,
                     builder.CreateString("W")),
        CreateTensor(builder, builder.CreateVector<int32_t>(bias_shape, 1),
                     TensorType_FLOAT32, /*buffer=*/2,
                     builder.CreateString("b")),
        CreateTensor(builder, builder.CreateVector<int32_t>(output_shape, 4),
                     TensorType_FLOAT32, /*buffer=*/0,
                     builder.CreateString("Y")),
    };

    const int32_t op_inputs[3] = {0, 1, 2};
    const int32_t op_outputs[1] = {3};

    flatbuffers::Offset<Operator> op = CreateOperator(
        builder, /*opcode_index=*/0,
        builder.CreateVector<int32_t>(op_inputs, 3),
        builder.CreateVector<int32_t>(op_outputs, 1),
        BuiltinOptions_DepthwiseConv2DOptions, depthwise_conv2d_options.Union(),
        /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS);

    int32_t subgraph_inputs[1] = {0};
    int32_t subgraph_outputs[1] = {3};
    flatbuffers::Offset<SubGraph> subgraph =
        CreateSubGraph(builder, builder.CreateVector(tensors, 4),
                       builder.CreateVector<int32_t>(subgraph_inputs, 1),
                       builder.CreateVector<int32_t>(subgraph_outputs, 1),
                       builder.CreateVector(&op, 1), /*name=*/0);

    flatbuffers::Offset<flatbuffers::String> description =
        builder.CreateString("DepthwiseConv2D model");

    flatbuffers::Offset<Model> model_buffer = CreateModel(
        builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
        builder.CreateVector(&subgraph, 1), description,
        builder.CreateVector(buffers, 3));

    builder.Finish(model_buffer);

    return std::vector<char>(builder.GetBufferPointer(),
                             builder.GetBufferPointer() + builder.GetSize());
  }

  int32_t batch_size_ = 1;
  int32_t groups_ = 1;
  int32_t depth_multiplier_ = 1;
  int32_t input_height_ = 1;
  int32_t input_width_ = 1;
  int32_t kernel_height_ = 1;
  int32_t kernel_width_ = 1;
  int32_t stride_height_ = 1;
  int32_t stride_width_ = 1;
  int32_t dilation_height_ = 1;
  int32_t dilation_width_ = 1;
};

}  // namespace

TEST(DepthwiseConv2D, 2x2) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto groups_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 32), std::ref(rng));

  DepthwiseConv2DTester()
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .Groups(groups_rng())
      .KernelHeight(2)
      .KernelWidth(2)
      .Test(xnnpack_delegate.get());
}

TEST(DepthwiseConv2D, 3x3) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto groups_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 32), std::ref(rng));

  DepthwiseConv2DTester()
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .Groups(groups_rng())
      .KernelHeight(3)
      .KernelWidth(3)
      .Test(xnnpack_delegate.get());
}

TEST(DepthwiseConv2D, SmallKernel) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 7), std::ref(rng));
  auto groups_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 32), std::ref(rng));

  DepthwiseConv2DTester()
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .Groups(groups_rng())
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .Test(xnnpack_delegate.get());
}

TEST(DepthwiseConv2D, Stride) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 5), std::ref(rng));
  auto stride_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto groups_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 32), std::ref(rng));

  DepthwiseConv2DTester()
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .Groups(groups_rng())
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .StrideHeight(stride_rng())
      .StrideWidth(stride_rng())
      .Test(xnnpack_delegate.get());
}

TEST(DepthwiseConv2D, Dilation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<int32_t>(10, 25), std::ref(rng));
  auto kernel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto dilation_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto group_rng =
      std::bind(std::uniform_int_distribution<int32_t>(3, 32), std::ref(rng));

  DepthwiseConv2DTester()
      .InputHeight(input_rng())
      .InputWidth(input_rng())
      .Groups(group_rng())
      .KernelHeight(kernel_rng())
      .KernelWidth(kernel_rng())
      .DilationHeight(dilation_rng())
      .DilationWidth(dilation_rng())
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
