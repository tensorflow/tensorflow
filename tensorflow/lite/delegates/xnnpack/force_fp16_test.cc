/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/binary_elementwise_tester.h"
#include "tensorflow/lite/delegates/xnnpack/conv_2d_tester.h"
#include "tensorflow/lite/delegates/xnnpack/fully_connected_tester.h"
#include "tensorflow/lite/delegates/xnnpack/pool_2d_tester.h"
#include "tensorflow/lite/delegates/xnnpack/unary_elementwise_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

TEST(ForceFp16, Add) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_ADD, delegate.get());
}

TEST(ForceFp16, Div) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_DIV, delegate.get());
}

TEST(ForceFp16, Maximum) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_MAXIMUM, delegate.get());
}

TEST(ForceFp16, Minimum) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_MINIMUM, delegate.get());
}

TEST(ForceFp16, Mul) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_MUL, delegate.get());
}

TEST(ForceFp16, SquaredDifference) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(3.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_SQUARED_DIFFERENCE, delegate.get());
}

TEST(ForceFp16, Sub) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  BinaryElementwiseTester()
      .Input1Shape({1, 4})
      .Input2Shape({1, 4})
      .RelativeTolerance(5.0e-2f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_SUB, delegate.get());
}

TEST(ForceFp16, Abs) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  UnaryElementwiseTester()
      .Shape({1, 4})
      .Tolerance({.relative = 1.0e-2f})
      .ExpectFp16Precision()
      .Test(BuiltinOperator_ABS, delegate.get());
}

TEST(ForceFp16, HardSwish) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  UnaryElementwiseTester()
      .Shape({1, 4})
      .Tolerance({.relative = 1.0e-2f})
      .ExpectFp16Precision()
      .Test(BuiltinOperator_HARD_SWISH, delegate.get());
}

TEST(ForceFp16, Neg) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  UnaryElementwiseTester()
      .Shape({1, 4})
      .Tolerance({.relative = 1.0e-2f})
      .ExpectFp16Precision()
      .Test(BuiltinOperator_NEG, delegate.get());
}

TEST(ForceFp16, GlobalAveragePooling2D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  Pool2DTester()
      .BatchSize(1)
      .InputHeight(4)
      .InputWidth(4)
      .Channels(4)
      .PoolingHeight(4)
      .PoolingWidth(4)
      .Tolerance(2.0e-3f)
      .ExpectFp16Precision()
      .Test(BuiltinOperator_AVERAGE_POOL_2D, delegate.get());
}

TEST(ForceFp16, FullyConnected) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  FullyConnectedTester()
      .InputShape({1, 4})
      .InputChannels(4)
      .OutputChannels(4)
      .FP16Weights()
      .NoBias()
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(delegate.get());
}

TEST(ForceFp16, Convolution) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  Conv2DTester()
      .BatchSize(1)
      .InputHeight(8)
      .InputWidth(8)
      .InputChannels(32)
      .OutputChannels(64)
      .KernelHeight(3)
      .KernelWidth(3)
      .RelativeTolerance(1.0e-2f)
      .ExpectFp16Precision()
      .Test(delegate.get());
}

// Helper function to invoke the interpreter and verify basic execution success
void TestFp16Delegate(const uint8_t* model_data) {
  const Model* model = GetModel(model_data);
  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
               TfLiteXNNPackDelegateDelete);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate.get()),
            kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
}

TEST(ForceFp16, SharedWeights) {
  // Two FullyConnected ops sharing the same weight tensor.
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_FULLY_CONNECTED)}};

  const std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
  const std::vector<float> bias_data = {1.0f, 1.0f};

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
      CreateBuffer(builder,
                   builder.CreateVector({})),  // Buffer 0 is always empty
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(filter_data.data()),
                       sizeof(float) * filter_data.size())),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(bias_data.data()),
                       sizeof(float) * bias_data.size()))};

  const std::vector<int32_t> filter_shape = {2, 2};
  const std::vector<int32_t> bias_shape = {2};
  const std::vector<int32_t> input_shape = {1, 2};

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          TensorType_FLOAT32),  // 0: Input
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(filter_shape.data(),
                                                 filter_shape.size()),
                   TensorType_FLOAT32, 1),  // 1: Weights
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(bias_shape.data(), bias_shape.size()),
          TensorType_FLOAT32, 2),  // 2: Bias
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          TensorType_FLOAT32),  // 3: Output1
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          TensorType_FLOAT32)  // 4: Output2
  };

  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  const std::array<int32_t, 3> op1_inputs{{0, 1, 2}};
  const std::array<int32_t, 1> op1_outputs{{3}};
  operators.emplace_back(CreateOperator(
      builder, 0,
      builder.CreateVector<int32_t>(op1_inputs.data(), op1_inputs.size()),
      builder.CreateVector<int32_t>(op1_outputs.data(), op1_outputs.size()),
      BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(builder).Union()));

  const std::array<int32_t, 3> op2_inputs{{0, 1, 2}};
  const std::array<int32_t, 1> op2_outputs{{4}};
  operators.emplace_back(CreateOperator(
      builder, 0,
      builder.CreateVector<int32_t>(op2_inputs.data(), op2_inputs.size()),
      builder.CreateVector<int32_t>(op2_outputs.data(), op2_outputs.size()),
      BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(builder).Union()));

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 2> subgraph_outputs{{3, 4}};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), builder.CreateString("SharedWeights"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);
  TestFp16Delegate(builder.GetBufferPointer());
}

TEST(ForceFp16, DuplicateInputs) {
  // An ADD op where LHS and RHS are exactly the same tensor index.
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_ADD)}};

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
      CreateBuffer(builder, builder.CreateVector({}))};

  const std::vector<int32_t> shape = {1, 4};
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),  // 0: Input
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32)  // 1: Output
  };

  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  const std::array<int32_t, 2> op_inputs{{0, 0}};  // Duplicate!
  const std::array<int32_t, 1> op_outputs{{1}};
  operators.emplace_back(CreateOperator(
      builder, 0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_AddOptions, CreateAddOptions(builder).Union()));

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{1}};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("DuplicateInputs"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);
  TestFp16Delegate(builder.GetBufferPointer());
}

TEST(ForceFp16, IntermediateOutput) {
  // Op1 output is used as Op2 input AND as a subgraph output.
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_ADD)}};

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
      CreateBuffer(builder, builder.CreateVector({}))};

  const std::vector<int32_t> shape = {1, 4};
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),  // 0: Input1
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),  // 1: Input2
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),  // 2: Intermediate
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32)  // 3: OutputFinal
  };

  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  const std::array<int32_t, 2> op1_inputs{{0, 1}};
  const std::array<int32_t, 1> op1_outputs{{2}};
  operators.emplace_back(CreateOperator(
      builder, 0,
      builder.CreateVector<int32_t>(op1_inputs.data(), op1_inputs.size()),
      builder.CreateVector<int32_t>(op1_outputs.data(), op1_outputs.size()),
      BuiltinOptions_AddOptions, CreateAddOptions(builder).Union()));

  const std::array<int32_t, 2> op2_inputs{{0, 2}};
  const std::array<int32_t, 1> op2_outputs{{3}};
  operators.emplace_back(CreateOperator(
      builder, 0,
      builder.CreateVector<int32_t>(op2_inputs.data(), op2_inputs.size()),
      builder.CreateVector<int32_t>(op2_outputs.data(), op2_outputs.size()),
      BuiltinOptions_AddOptions, CreateAddOptions(builder).Union()));

  const std::array<int32_t, 2> subgraph_inputs{{0, 1}};
  const std::array<int32_t, 2> subgraph_outputs{{2, 3}};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("IntermediateOutput"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);
  TestFp16Delegate(builder.GetBufferPointer());
}

}  // namespace xnnpack
}  // namespace tflite
