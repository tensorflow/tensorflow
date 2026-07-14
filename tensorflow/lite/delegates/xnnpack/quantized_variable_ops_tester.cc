/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/quantized_variable_ops_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

namespace {
// This is the order we declare the operators in each model, it is the same for
// all models in this test.
constexpr uint32_t VAR_HANDLE = 0;
constexpr uint32_t READ_VARIABLE = 1;
constexpr uint32_t ASSIGN_VARIABLE = 2;
constexpr uint32_t CALL_ONCE = 3;
}  // namespace

std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
NewXnnPackDelegateSupportingVariableOps() {
  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&options),
                       TfLiteXNNPackDelegateDelete);
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  return xnnpack_delegate;
}

std::vector<char> QuantizedVariableOpsTester::CreateModelAssignThenRead()
    const {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };

  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  // tensor 0 is graph input
  // tensor 1 is VAR_HANDLE output
  // tensor 2 is graph output
  const std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("shared_name"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 0}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> read_op =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({1}),
                     builder.CreateVector<int32_t>({2}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({2}),
      builder.CreateVector({var_handle_op, assign_op, read_op}));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char> QuantizedVariableOpsTester::CreateModelAssignTwiceThenRead()
    const {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };

  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  // tensor 0 is graph input
  // tensor 1 is VAR_HANDLE output
  // tensor 2 is graph output
  // tensor 3 is second graph input, initial values
  const std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("shared_name"))
          .Union());

  const flatbuffers::Offset<Operator> initial_assign_op = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 3}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> assign_op = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 0}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> read_op =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({1}),
                     builder.CreateVector<int32_t>({2}));

  const std::array<flatbuffers::Offset<Operator>, 4> ops = {
      var_handle_op,
      initial_assign_op,
      assign_op,
      read_op,
  };

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({0, 3}), builder.CreateVector<int32_t>({2}),
      builder.CreateVector(ops.data(), ops.size()));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char>
QuantizedVariableOpsTester::CreateModelAssignThenReadUsingAnotherVarHandle()
    const {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };

  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  // tensor 0 is graph input
  // tensor 1 is VAR_HANDLE output
  // tensor 2 is graph output
  // tensor 3 is VAR_HANDLE output used by the READ_VARIABLE
  const std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
  }};

  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("shared_name"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 0}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> var_handle_op_for_read = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({3}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("shared_name"))
          .Union());

  const flatbuffers::Offset<Operator> read_op =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({3}),
                     builder.CreateVector<int32_t>({2}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({2}),
      builder.CreateVector(
          {var_handle_op, assign_op, var_handle_op_for_read, read_op}));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char>
QuantizedVariableOpsTester::CreateModelTwoVarHandlesAssignThenRead() const {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };

  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  // tensors 0 and 1 are graph inputs
  // tensors 2 and 3 are graph outputs
  // tensors 4 and 5 are VAR_HANDLE outputs
  const std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
  }};

  const flatbuffers::Offset<Operator> var_handle_op_1 = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({4}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name1"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_1 = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({4, 0}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> read_op_1 =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({4}),
                     builder.CreateVector<int32_t>({2}));

  const flatbuffers::Offset<Operator> var_handle_op_2 = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({5}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name2"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_2 = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({5, 1}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> read_op_2 =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({5}),
                     builder.CreateVector<int32_t>({3}));

  const std::array<flatbuffers::Offset<Operator>, 6> ops = {
      var_handle_op_1, assign_op_1, read_op_1,
      var_handle_op_2, assign_op_2, read_op_2,
  };

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({0, 1}),
      builder.CreateVector<int32_t>({2, 3}),
      builder.CreateVector(ops.data(), ops.size()));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char>
QuantizedVariableOpsTester::CreateModelTwoSubgraphsReadAssign() const {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };

  const uint32_t buffer1_id = 1;
  const std::vector<uint8_t> buffer_data1(InputSize(), 3);
  const uint32_t buffer2_id = 2;
  const std::vector<uint8_t> buffer_data2(InputSize(), 2);
  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data1.data()),
                       buffer_data1.size())),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data2.data()),
                       buffer_data2.size())),
  }};

  // tensor 0 is primary graph VAR_HANDLE 1 output
  // tensor 1 is primary graph VAR_HANDLE 2 output
  // tensor 2 is graph output 1
  // tensor 3 is graph output 2
  const std::vector<flatbuffers::Offset<Tensor>> primary_tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  // Operators in primary graph.
  const flatbuffers::Offset<Operator> call_once_op = CreateOperator(
      builder, CALL_ONCE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({}), tflite::BuiltinOptions_CallOnceOptions,
      CreateCallOnceOptions(builder, 1).Union());

  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name1"))
          .Union());

  const flatbuffers::Offset<Operator> read_op =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({0}),
                     builder.CreateVector<int32_t>({2}));

  const flatbuffers::Offset<Operator> var_handle_op2 = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name2"))
          .Union());

  const flatbuffers::Offset<Operator> read_op2 =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({1}),
                     builder.CreateVector<int32_t>({3}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(primary_tensors.data(), primary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({2, 3}),
      builder.CreateVector({
          call_once_op,
          var_handle_op,
          read_op,
          var_handle_op2,
          read_op2,
      }));

  // tensor 0 is secondary graph VAR_HANDLE 2 output
  // tensor 1 is secondary graph VAR_HANDLE 1 output
  // tensor 2 is secondary graph values to be assigned, this is buffer 1
  // tensor 3 is secondary graph values to be assigned, this is buffer 2
  const std::vector<flatbuffers::Offset<Tensor>> secondary_tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/buffer1_id, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/buffer2_id, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  // Operators in secondary graph.
  const flatbuffers::Offset<Operator> var_handle_op_secondary = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name2"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_secondary = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({0, 2}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> var_handle_op_secondary2 = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name1"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_secondary2 = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 3}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<SubGraph> secondary_subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(secondary_tensors.data(), secondary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({}),
      builder.CreateVector({
          var_handle_op_secondary,
          assign_op_secondary,
          var_handle_op_secondary2,
          assign_op_secondary2,
      }));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector({subgraph, secondary_subgraph}),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char>
QuantizedVariableOpsTester::CreateModelTwoSubgraphsReadAssignOneVarHandle()
    const {
  flatbuffers::FlatBufferBuilder builder;

  const std::array<flatbuffers::Offset<OperatorCode>, 4> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };

  const uint32_t buffer1_id = 1;
  const std::vector<uint8_t> buffer_data1(InputSize(), 3);
  const uint32_t buffer2_id = 2;
  const std::vector<uint8_t> buffer_data2(InputSize(), 2);
  const std::array<flatbuffers::Offset<Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data1.data()),
                       buffer_data1.size())),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data2.data()),
                       buffer_data2.size())),
  }};

  // tensor 0 is primary graph VAR_HANDLE 1 output
  // tensor 1 is graph output 1
  const std::array<flatbuffers::Offset<Tensor>, 2> primary_tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  // Operators in primary graph.
  const flatbuffers::Offset<Operator> call_once_op = CreateOperator(
      builder, CALL_ONCE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({}), tflite::BuiltinOptions_CallOnceOptions,
      CreateCallOnceOptions(builder, 1).Union());

  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name1"))
          .Union());

  const flatbuffers::Offset<Operator> read_op =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({0}),
                     builder.CreateVector<int32_t>({1}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(primary_tensors.data(), primary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({1}),
      builder.CreateVector({
          call_once_op,
          var_handle_op,
          read_op,
      }));

  // tensor 0 is secondary graph VAR_HANDLE 2 output
  // tensor 1 is secondary graph VAR_HANDLE 1 output
  // tensor 2 is secondary graph values to be assigned, this is buffer 1
  // tensor 3 is secondary graph values to be assigned, this is buffer 2
  const std::array<flatbuffers::Offset<Tensor>, 4> secondary_tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/buffer1_id, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/buffer2_id, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  // Operators in secondary graph.
  const flatbuffers::Offset<Operator> var_handle_op_secondary = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name2"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_secondary = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({0, 2}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> var_handle_op_secondary2 = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name1"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_secondary2 = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 3}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<SubGraph> secondary_subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(secondary_tensors.data(), secondary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({}),
      builder.CreateVector({
          var_handle_op_secondary,
          assign_op_secondary,
          var_handle_op_secondary2,
          assign_op_secondary2,
      }));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector({subgraph, secondary_subgraph}),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char>
QuantizedVariableOpsTester::CreateModelTwoSubgraphsReadAssignOneVarHandle2()
    const {
  flatbuffers::FlatBufferBuilder builder;

  const std::array<flatbuffers::Offset<OperatorCode>, 4> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };

  const uint32_t buffer1_id = 1;
  const std::vector<uint8_t> buffer_data1(InputSize(), 3);
  const uint32_t buffer2_id = 2;
  const std::vector<uint8_t> buffer_data2(InputSize(), 2);
  const std::array<flatbuffers::Offset<Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data1.data()),
                       buffer_data1.size())),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data2.data()),
                       buffer_data2.size())),
  }};

  // tensor 0 is primary graph VAR_HANDLE 2 output
  // tensor 1 is graph output 1
  const std::array<flatbuffers::Offset<Tensor>, 2> primary_tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  // Operators in primary graph.
  const flatbuffers::Offset<Operator> call_once_op = CreateOperator(
      builder, CALL_ONCE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({}), tflite::BuiltinOptions_CallOnceOptions,
      CreateCallOnceOptions(builder, 1).Union());

  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name2"))
          .Union());

  const flatbuffers::Offset<Operator> read_op =
      CreateOperator(builder, READ_VARIABLE, builder.CreateVector<int32_t>({0}),
                     builder.CreateVector<int32_t>({1}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(primary_tensors.data(), primary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({1}),
      builder.CreateVector({
          call_once_op,
          var_handle_op,
          read_op,
      }));

  // tensor 0 is secondary graph VAR_HANDLE 1 output
  // tensor 1 is secondary graph VAR_HANDLE 2 output
  // tensor 2 is secondary graph values to be assigned, this is buffer 1
  // tensor 3 is secondary graph values to be assigned, this is buffer 2
  const std::array<flatbuffers::Offset<Tensor>, 4> secondary_tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/buffer1_id, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/buffer2_id, /*name*/ 0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  // Operators in secondary graph.
  const flatbuffers::Offset<Operator> var_handle_op_secondary = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name1"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_secondary = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({0, 2}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<Operator> var_handle_op_secondary2 = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("name2"))
          .Union());

  const flatbuffers::Offset<Operator> assign_op_secondary2 = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({1, 3}),
      builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<SubGraph> secondary_subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(secondary_tensors.data(), secondary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({}),
      builder.CreateVector({
          var_handle_op_secondary,
          assign_op_secondary,
          var_handle_op_secondary2,
          assign_op_secondary2,
      }));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector({subgraph, secondary_subgraph}),
      builder.CreateString("ReadVariable model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

void QuantizedVariableOpsTester::TestAssignThenRead(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model = CreateModelAssignThenRead();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

void QuantizedVariableOpsTester::TestAssignTwiceThenRead(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model = CreateModelAssignTwiceThenRead();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

void QuantizedVariableOpsTester::TestAssignThenReadUsingAnotherVarHandle(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model =
      CreateModelAssignThenReadUsingAnotherVarHandle();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

void QuantizedVariableOpsTester::TestTwoVarHandlesAssignThenRead(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model = CreateModelTwoVarHandlesAssignThenRead();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

void QuantizedVariableOpsTester::TestTwoSubgraphsReadAssign(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model = CreateModelTwoSubgraphsReadAssign();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

void QuantizedVariableOpsTester::TestTwoSubgraphsReadAssignOneVarHandle(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model =
      CreateModelTwoSubgraphsReadAssignOneVarHandle();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

void QuantizedVariableOpsTester::TestTwoSubgraphsReadAssignOneVarHandle2(
    TfLiteDelegate* delegate) const {
  const std::vector<char> model =
      CreateModelTwoSubgraphsReadAssignOneVarHandle2();
  if (Unsigned()) {
    Test<uint8_t>(delegate, model);
  } else {
    Test<int8_t>(delegate, model);
  }
}

template <class T>
void QuantizedVariableOpsTester::Test(TfLiteDelegate* delegate,
                                      const std::vector<char>& buffer) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_int_distribution<T>(std::numeric_limits<T>::min(),
                                                 std::numeric_limits<T>::max()),
                std::ref(rng));
  delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;

  const Model* model = GetModel(buffer.data());
  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), NumInputs());
  ASSERT_EQ(default_interpreter->inputs().size(), NumInputs());

  ASSERT_EQ(delegate_interpreter->outputs().size(), NumOutputs());
  ASSERT_EQ(default_interpreter->outputs().size(), NumOutputs());

  ASSERT_EQ(delegate_interpreter->subgraphs_size(), NumSubgraphs());
  ASSERT_EQ(default_interpreter->subgraphs_size(), NumSubgraphs());

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  for (size_t i = 0; i < NumInputs(); i++) {
    T* default_input_data = default_interpreter->typed_input_tensor<T>(i);
    std::generate_n(default_input_data, InputSize(), std::ref(input_rng));
    T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(i);
    std::copy_n(default_input_data, InputSize(), delegate_input_data);
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  for (size_t i = 0; i < NumOutputs(); i++) {
    const T* default_output_data =
        default_interpreter->typed_output_tensor<T>(i);
    const T* delegate_output_data =
        delegate_interpreter->typed_output_tensor<T>(i);
    for (size_t i = 0; i < OutputSize(); i++) {
      EXPECT_EQ(delegate_output_data[i], default_output_data[i]);
    }
  }
}

}  // namespace xnnpack
}  // namespace tflite
