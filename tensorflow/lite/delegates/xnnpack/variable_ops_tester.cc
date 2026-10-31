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

#include "tensorflow/lite/delegates/xnnpack/variable_ops_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/core/c/common.h"
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
  return xnnpack_delegate;
}

std::vector<char> VariableOpsTester::CreateModelAssignOnly() const {
  constexpr int32_t kInputTensorIndex = 0;
  constexpr int32_t kVarHandleTensorIndex = 1;
  constexpr size_t kNumSubgraphs = 1;
  constexpr char kContainerName[] = "container";
  constexpr char kSharedName[] = "shared_name";
  constexpr char kModelDescription[] = "AssignVariable model";

  flatbuffers::FlatBufferBuilder builder;

  // Keep the same operator code indices (VAR_HANDLE, READ_VARIABLE,
  // ASSIGN_VARIABLE) used across all VariableOpsTester models.
  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };

  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  // Tensor 0 is the graph input; tensor 1 is the VAR_HANDLE resource output.
  const std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
  }};

  // Create VAR_HANDLE and ASSIGN_VARIABLE operators with no subgraph outputs.
  const flatbuffers::Offset<Operator> var_handle_op = CreateOperator(
      builder, /*opcode_index=*/VAR_HANDLE,
      /*inputs=*/builder.CreateVector<int32_t>({}),
      /*outputs=*/builder.CreateVector<int32_t>({kVarHandleTensorIndex}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString(kContainerName),
                             builder.CreateString(kSharedName))
          .Union());

  const flatbuffers::Offset<Operator> assign_op = CreateOperator(
      builder, /*opcode_index=*/ASSIGN_VARIABLE,
      /*inputs=*/
      builder.CreateVector<int32_t>({kVarHandleTensorIndex, kInputTensorIndex}),
      /*outputs=*/builder.CreateVector<int32_t>({}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      /*inputs=*/builder.CreateVector<int32_t>({kInputTensorIndex}),
      /*outputs=*/builder.CreateVector<int32_t>({}),
      builder.CreateVector({var_handle_op, assign_op}));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, /*version=*/TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, /*len=*/kNumSubgraphs),
      builder.CreateString(kModelDescription),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char> VariableOpsTester::CreateModelAssignThenRead() const {
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
          TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
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

std::vector<char> VariableOpsTester::CreateModelAssignTwiceThenRead() const {
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
          TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
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
VariableOpsTester::CreateModelAssignThenReadUsingAnotherVarHandle() const {
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
          TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(ResourceShape().data(),
                                                 ResourceShape().size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
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

std::vector<char> VariableOpsTester::CreateModelTwoVarHandlesAssignThenRead()
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

  // tensors 0 and 1 are graph inputs
  // tensors 2 and 3 are graph outputs
  // tensors 4 and 5 are VAR_HANDLE outputs
  const std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
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

std::vector<char> VariableOpsTester::CreateModelTwoSubgraphsReadAssign() const {
  flatbuffers::FlatBufferBuilder builder;

  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };

  const uint32_t buffer1_id = 1;
  const std::vector<float> buffer_data1(InputSize(), 3.14f);
  const uint32_t buffer2_id = 2;
  const std::vector<float> buffer_data2(InputSize(), 2.71f);
  const std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data1.data()),
                       buffer_data1.size() * sizeof(float) / sizeof(uint8_t))),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data2.data()),
                       buffer_data2.size() * sizeof(float) / sizeof(uint8_t))),
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
          TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
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
          TensorType_FLOAT32, buffer1_id),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32, buffer2_id),
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
VariableOpsTester::CreateModelTwoSubgraphsReadAssignOneVarHandle() const {
  flatbuffers::FlatBufferBuilder builder;

  const std::array<flatbuffers::Offset<OperatorCode>, 4> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };

  const uint32_t buffer1_id = 1;
  const std::vector<float> buffer_data1(InputSize(), 3.14f);
  const uint32_t buffer2_id = 2;
  const std::vector<float> buffer_data2(InputSize(), 2.71f);
  const std::array<flatbuffers::Offset<Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data1.data()),
                       buffer_data1.size() * sizeof(float) / sizeof(uint8_t))),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data2.data()),
                       buffer_data2.size() * sizeof(float) / sizeof(uint8_t))),
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
          TensorType_FLOAT32),
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
          TensorType_FLOAT32, buffer1_id),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32, buffer2_id),
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
VariableOpsTester::CreateModelTwoSubgraphsReadAssignOneVarHandle2() const {
  flatbuffers::FlatBufferBuilder builder;

  const std::array<flatbuffers::Offset<OperatorCode>, 4> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };

  const uint32_t buffer1_id = 1;
  const std::vector<float> buffer_data1(InputSize(), 3.14f);
  const uint32_t buffer2_id = 2;
  const std::vector<float> buffer_data2(InputSize(), 2.71f);
  const std::array<flatbuffers::Offset<Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data1.data()),
                       buffer_data1.size() * sizeof(float) / sizeof(uint8_t))),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(buffer_data2.data()),
                       buffer_data2.size() * sizeof(float) / sizeof(uint8_t))),
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
          TensorType_FLOAT32),
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
          TensorType_FLOAT32, buffer1_id),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32, buffer2_id),
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

void VariableOpsTester::Test(TfLiteDelegate* delegate,
                             const std::vector<char>& buffer) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
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
    float* default_input_data =
        default_interpreter->typed_input_tensor<float>(i);
    std::generate_n(default_input_data, InputSize(), std::ref(f32rng));
    float* delegate_input_data =
        delegate_interpreter->typed_input_tensor<float>(i);
    std::copy_n(default_input_data, InputSize(), delegate_input_data);
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  for (size_t i = 0; i < NumOutputs(); i++) {
    const float* default_output_data =
        default_interpreter->typed_output_tensor<float>(i);
    const float* delegate_output_data =
        delegate_interpreter->typed_output_tensor<float>(i);
    for (size_t i = 0; i < OutputSize(); i++) {
      EXPECT_EQ(delegate_output_data[i], default_output_data[i]);
    }
  }
}

void VariableOpsTester::TestResizeAssignOnly(TfLiteDelegate* delegate) const {
  const std::vector<char> buffer = CreateModelAssignOnly();
  std::mt19937 rng(42);
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
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
  ASSERT_TRUE(delegate_interpreter);
  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  float* in0 = delegate_interpreter->typed_input_tensor<float>(0);
  std::generate_n(in0, InputSize(), std::ref(f32rng));
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  const std::vector<int> grown_shape = {1, 5, 2, 3};
  const size_t grown_elements = 1 * 5 * 2 * 3;
  ASSERT_EQ(delegate_interpreter->ResizeInputTensor(
                delegate_interpreter->inputs()[0], grown_shape),
            kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  float* grown_in = delegate_interpreter->typed_input_tensor<float>(0);
  std::generate_n(grown_in, grown_elements, std::ref(f32rng));
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  const std::vector<int> shrunk_shape = {1, 1, 2, 3};
  const size_t shrunk_elements = 1 * 1 * 2 * 3;
  ASSERT_EQ(delegate_interpreter->ResizeInputTensor(
                delegate_interpreter->inputs()[0], shrunk_shape),
            kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  float* shrunk_in = delegate_interpreter->typed_input_tensor<float>(0);
  std::generate_n(shrunk_in, shrunk_elements, std::ref(f32rng));
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
}

void VariableOpsTester::TestResizeAssignThenRead(
    TfLiteDelegate* delegate) const {
  const std::vector<char> buffer = CreateModelAssignThenRead();
  std::mt19937 rng(1337);
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
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
  ASSERT_TRUE(delegate_interpreter);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  std::vector<float> expected(InputSize());
  std::generate(expected.begin(), expected.end(), std::ref(f32rng));
  std::copy(expected.begin(), expected.end(),
            delegate_interpreter->typed_input_tensor<float>(0));
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(delegate_interpreter->typed_output_tensor<float>(0)[i],
              expected[i]);
  }

  for (const std::vector<int>& next_shape :
       std::vector<std::vector<int>>{{1, 5, 2, 3}, {1, 1, 2, 3}}) {
    const size_t next_elems = static_cast<size_t>(next_shape[0]) *
                              next_shape[1] * next_shape[2] * next_shape[3];
    ASSERT_EQ(delegate_interpreter->ResizeInputTensor(
                  delegate_interpreter->inputs()[0], next_shape),
              kTfLiteOk);
    ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(
        delegate_interpreter->tensor(delegate_interpreter->outputs()[0])->bytes,
        next_elems * sizeof(float));

    expected.resize(next_elems);
    std::generate(expected.begin(), expected.end(), std::ref(f32rng));
    std::copy(expected.begin(), expected.end(),
              delegate_interpreter->typed_input_tensor<float>(0));
    ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
    for (size_t i = 0; i < next_elems; ++i) {
      EXPECT_EQ(delegate_interpreter->typed_output_tensor<float>(0)[i],
                expected[i]);
    }
  }
}

void VariableOpsTester::TestRepeatedResizeCyclesHaveNoLeaksOrStaleCapacity(
    TfLiteDelegate* delegate) const {
  const std::vector<char> buffer = CreateModelAssignThenRead();
  std::mt19937 rng(20261031);
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-4.0f, 4.0f),
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
  ASSERT_TRUE(delegate_interpreter);
  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  const std::vector<std::vector<int>> shapes = {
      {1, 1, 1, 1}, {2, 7, 5, 3}, {1, 3, 3, 2},
      {4, 9, 4, 3}, {4, 9, 4, 3}, {1, 1, 2, 1},
  };
  std::vector<float> expected;
  for (int cycle = 0; cycle < 32; ++cycle) {
    const std::vector<int>& shape = shapes[cycle % shapes.size()];
    const size_t elems =
        static_cast<size_t>(shape[0]) * shape[1] * shape[2] * shape[3];
    ASSERT_EQ(delegate_interpreter->ResizeInputTensor(
                  delegate_interpreter->inputs()[0], shape),
              kTfLiteOk);
    ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
    expected.resize(elems);
    std::generate(expected.begin(), expected.end(), std::ref(f32rng));
    std::copy(expected.begin(), expected.end(),
              delegate_interpreter->typed_input_tensor<float>(0));
    ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
    const float* actual = delegate_interpreter->typed_output_tensor<float>(0);
    for (size_t i = 0; i < elems; ++i) {
      ASSERT_EQ(actual[i], expected[i]) << "cycle=" << cycle << " idx=" << i;
    }
  }
}

const char* VariableOpGuardCaseName(VariableOpGuardCase test_case) {
  switch (test_case) {
    case VariableOpGuardCase::kNullContextInPrepare:
      return "NullContextInPrepare";
    case VariableOpGuardCase::kNullExecutionPlanInPrepare:
      return "NullExecutionPlanInPrepare";
    case VariableOpGuardCase::kNullParamsInSubgraphCreate:
      return "NullParamsInSubgraphCreate";
    case VariableOpGuardCase::kNegativeInputTensorsSizeInSubgraphCreate:
      return "NegativeInputTensorsSizeInSubgraphCreate";
    case VariableOpGuardCase::kOutOfBoundsOutputTensorIndexInSubgraphCreate:
      return "OutOfBoundsOutputTensorIndexInSubgraphCreate";
    case VariableOpGuardCase::kNullNodeInputsVarHandle:
      return "NullNodeInputsVarHandle";
    case VariableOpGuardCase::kNullNodeInputsAssignVariable:
      return "NullNodeInputsAssignVariable";
    case VariableOpGuardCase::kNullNodeInputsReadVariable:
      return "NullNodeInputsReadVariable";
    case VariableOpGuardCase::kNullNodeOutputsVarHandle:
      return "NullNodeOutputsVarHandle";
    case VariableOpGuardCase::kNullNodeOutputsAssignVariable:
      return "NullNodeOutputsAssignVariable";
    case VariableOpGuardCase::kNullNodeOutputsReadVariable:
      return "NullNodeOutputsReadVariable";
    case VariableOpGuardCase::kNegativeInputsSizeVarHandle:
      return "NegativeInputsSizeVarHandle";
    case VariableOpGuardCase::kUnexpectedInputArityVarHandle:
      return "UnexpectedInputArityVarHandle";
    case VariableOpGuardCase::kUnexpectedOutputArityAssignVariable:
      return "UnexpectedOutputArityAssignVariable";
    case VariableOpGuardCase::kNullVarHandleBuiltinData:
      return "NullVarHandleBuiltinData";
    case VariableOpGuardCase::kVarHandleOutputNegativeOne:
      return "VarHandleOutputNegativeOne";
    case VariableOpGuardCase::kVarHandleOutputNegativeTwo:
      return "VarHandleOutputNegativeTwo";
    case VariableOpGuardCase::kVarHandleOutputOutOfBounds:
      return "VarHandleOutputOutOfBounds";
    case VariableOpGuardCase::kAssignResourceInputNegativeOne:
      return "AssignResourceInputNegativeOne";
    case VariableOpGuardCase::kAssignResourceInputNegativeTwo:
      return "AssignResourceInputNegativeTwo";
    case VariableOpGuardCase::kAssignResourceInputOutOfBounds:
      return "AssignResourceInputOutOfBounds";
    case VariableOpGuardCase::kAssignValueInputNegativeOne:
      return "AssignValueInputNegativeOne";
    case VariableOpGuardCase::kAssignValueInputNegativeTwo:
      return "AssignValueInputNegativeTwo";
    case VariableOpGuardCase::kAssignValueInputOutOfBounds:
      return "AssignValueInputOutOfBounds";
    case VariableOpGuardCase::kReadResourceInputNegativeOne:
      return "ReadResourceInputNegativeOne";
    case VariableOpGuardCase::kReadResourceInputNegativeTwo:
      return "ReadResourceInputNegativeTwo";
    case VariableOpGuardCase::kReadResourceInputOutOfBounds:
      return "ReadResourceInputOutOfBounds";
    case VariableOpGuardCase::kReadValueOutputNegativeOne:
      return "ReadValueOutputNegativeOne";
    case VariableOpGuardCase::kReadValueOutputNegativeTwo:
      return "ReadValueOutputNegativeTwo";
    case VariableOpGuardCase::kReadValueOutputOutOfBounds:
      return "ReadValueOutputOutOfBounds";
    case VariableOpGuardCase::kWrongResourceTensorType:
      return "WrongResourceTensorType";
    case VariableOpGuardCase::kAssignInputNullDims:
      return "AssignInputNullDims";
    case VariableOpGuardCase::kAssignInputScalarRankZero:
      return "AssignInputScalarRankZero";
    case VariableOpGuardCase::kAssignInputNegativeDim:
      return "AssignInputNegativeDim";
    case VariableOpGuardCase::kAssignInputExcessiveRank:
      return "AssignInputExcessiveRank";
    case VariableOpGuardCase::kAssignInputByteSizeOverflow:
      return "AssignInputByteSizeOverflow";
    case VariableOpGuardCase::kAssignInputShapeExceedsTensorBytes:
      return "AssignInputShapeExceedsTensorBytes";
    case VariableOpGuardCase::kReadOutputNullDims:
      return "ReadOutputNullDims";
    case VariableOpGuardCase::kReadOutputNegativeDim:
      return "ReadOutputNegativeDim";
    case VariableOpGuardCase::kReadOutputExcessiveRank:
      return "ReadOutputExcessiveRank";
    case VariableOpGuardCase::kReadOutputByteSizeOverflow:
      return "ReadOutputByteSizeOverflow";
    case VariableOpGuardCase::kReadOutputShapeExceedsTensorBytes:
      return "ReadOutputShapeExceedsTensorBytes";
    case VariableOpGuardCase::kConflictingVarHandlesOnSameLocalResourceTensor:
      return "ConflictingVarHandlesOnSameLocalResourceTensor";
    case VariableOpGuardCase::kExternalOutputBytesSmallerThanXnnpackShapeAtInvoke:
      return "ExternalOutputBytesSmallerThanXnnpackShapeAtInvoke";
    case VariableOpGuardCase::kExternalNullDataWithNonZeroXnnpackShapeAtInvoke:
      return "ExternalNullDataWithNonZeroXnnpackShapeAtInvoke";
    case VariableOpGuardCase::kReshapedExternalInputBytesSmallerThanShapeInPrepare:
      return "ReshapedExternalInputBytesSmallerThanShapeInPrepare";
    case VariableOpGuardCase::kCreateReshapeZeroInputs:
      return "CreateReshapeZeroInputs";
    case VariableOpGuardCase::kCreateReshapeOutOfBoundsInput:
      return "CreateReshapeOutOfBoundsInput";
    case VariableOpGuardCase::kCreateReshapeOutOfBoundsShapeInput:
      return "CreateReshapeOutOfBoundsShapeInput";
    case VariableOpGuardCase::kCreateMeanZeroInputs:
      return "CreateMeanZeroInputs";
    case VariableOpGuardCase::kCreatePadOutOfBoundsPaddingsInput:
      return "CreatePadOutOfBoundsPaddingsInput";
    case VariableOpGuardCase::kCreateResizeBilinearZeroInputs:
      return "CreateResizeBilinearZeroInputs";
    case VariableOpGuardCase::kCreateSliceLackingInputs:
      return "CreateSliceLackingInputs";
    case VariableOpGuardCase::kCreateStridedSliceLackingInputs:
      return "CreateStridedSliceLackingInputs";
    case VariableOpGuardCase::kCreateSplitMissingInputSlot1:
      return "CreateSplitMissingInputSlot1";
    case VariableOpGuardCase::kCreateSplitOutOfBoundsValueInput:
      return "CreateSplitOutOfBoundsValueInput";
    case VariableOpGuardCase::kCreateSplitOutOfBoundsAxisInput:
      return "CreateSplitOutOfBoundsAxisInput";
    case VariableOpGuardCase::kCreateTransposeZeroInputs:
      return "CreateTransposeZeroInputs";
    case VariableOpGuardCase::kCreateTransposeOutOfBoundsInput:
      return "CreateTransposeOutOfBoundsInput";
    case VariableOpGuardCase::kCreateTransposeOutOfBoundsPermInput:
      return "CreateTransposeOutOfBoundsPermInput";
    case VariableOpGuardCase::kCreateTransposeConvOutOfBoundsFilterInput:
      return "CreateTransposeConvOutOfBoundsFilterInput";
    case VariableOpGuardCase::kCreateTransposeConvOutOfBoundsBiasInput:
      return "CreateTransposeConvOutOfBoundsBiasInput";
    case VariableOpGuardCase::kCreateTransposeConvOutOfBoundsOutputShapeInput:
      return "CreateTransposeConvOutOfBoundsOutputShapeInput";
    case VariableOpGuardCase::kCreateSliceZeroOutputs:
      return "CreateSliceZeroOutputs";
    case VariableOpGuardCase::kCreateStridedSliceZeroOutputs:
      return "CreateStridedSliceZeroOutputs";
    case VariableOpGuardCase::kCreateBatchMatMulZeroInputs:
      return "CreateBatchMatMulZeroInputs";
    case VariableOpGuardCase::kCreateBatchMatMulZeroOutputs:
      return "CreateBatchMatMulZeroOutputs";
    case VariableOpGuardCase::kCreateSdpaCustomZeroInputs:
      return "CreateSdpaCustomZeroInputs";
    case VariableOpGuardCase::kCreateSdpaCustomZeroOutputs:
      return "CreateSdpaCustomZeroOutputs";
    case VariableOpGuardCase::kCreateOptionalNegativeInputInDefaultUnaryOp:
      return "CreateOptionalNegativeInputInDefaultUnaryOp";
    case VariableOpGuardCase::kCreateOptionalNegativeOutputInSubgraphCreate:
      return "CreateOptionalNegativeOutputInSubgraphCreate";
    case VariableOpGuardCase::kCreateLeakyReluNullBuiltinData:
      return "CreateLeakyReluNullBuiltinData";
    case VariableOpGuardCase::kCreateGeluNullBuiltinData:
      return "CreateGeluNullBuiltinData";
    case VariableOpGuardCase::kCreateSdpaCustomNegativeInitialDataSize:
      return "CreateSdpaCustomNegativeInitialDataSize";
  }
  return "UnknownGuardCase";
}

const char* MalformedVariableOpModelCaseName(
    MalformedVariableOpModelCase test_case) {
  switch (test_case) {
    case MalformedVariableOpModelCase::kVarHandleZeroOutputs:
      return "VarHandleZeroOutputs";
    case MalformedVariableOpModelCase::kReadVariableZeroInputs:
      return "ReadVariableZeroInputs";
    case MalformedVariableOpModelCase::kReadVariableZeroOutputs:
      return "ReadVariableZeroOutputs";
    case MalformedVariableOpModelCase::kAssignVariableZeroInputs:
      return "AssignVariableZeroInputs";
    case MalformedVariableOpModelCase::kAssignVariableOneInput:
      return "AssignVariableOneInput";
    case MalformedVariableOpModelCase::kAssignVariableUnexpectedOutput:
      return "AssignVariableUnexpectedOutput";
    case MalformedVariableOpModelCase::kAssignVariableNegativeValueInput:
      return "AssignVariableNegativeValueInput";
    case MalformedVariableOpModelCase::kAssignVariableOutOfBoundsValueInput:
      return "AssignVariableOutOfBoundsValueInput";
    case MalformedVariableOpModelCase::kReadVariableNegativeValueOutput:
      return "ReadVariableNegativeValueOutput";
    case MalformedVariableOpModelCase::kReadVariableOutOfBoundsValueOutput:
      return "ReadVariableOutOfBoundsValueOutput";
    case MalformedVariableOpModelCase::
        kVarHandleAndReadVariableNegativeResourceWire:
      return "VarHandleAndReadVariableNegativeResourceWire";
    case MalformedVariableOpModelCase::
        kVarHandleAndAssignVariableNegativeResourceWire:
      return "VarHandleAndAssignVariableNegativeResourceWire";
  }
  return "UnknownMalformedCase";
}

void VariableOpsTester::TestGuardRejectCase(
    VariableOpGuardCase test_case) const {
  auto make_interp_from_buf = [](const std::vector<char>& buf,
                                 std::vector<char>* backing)
      -> std::unique_ptr<Interpreter> {
    *backing = buf;
    const Model* model = GetModel(backing->data());
    std::unique_ptr<Interpreter> interp;
    if (InterpreterBuilder(
            model,
            ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
            &interp) != kTfLiteOk) {
      return nullptr;
    }
    if (interp->AllocateTensors() != kTfLiteOk) {
      return nullptr;
    }
    return interp;
  };

  auto make_assign_then_read_interp =
      [&](std::vector<char>* backing) -> std::unique_ptr<Interpreter> {
    VariableOpsTester t;
    t.NumInputs(1).NumOutputs(1);
    return make_interp_from_buf(t.CreateModelAssignThenRead(), backing);
  };

  auto make_assign_only_interp =
      [&](std::vector<char>* backing) -> std::unique_ptr<Interpreter> {
    VariableOpsTester t;
    t.NumInputs(1).NumOutputs(0);
    return make_interp_from_buf(t.CreateModelAssignOnly(), backing);
  };

  auto make_read_only_interp =
      [&](std::vector<char>* backing) -> std::unique_ptr<Interpreter> {
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
        CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
    };
    const std::vector<flatbuffers::Offset<Buffer>> buffers = {
        CreateBuffer(builder, builder.CreateVector({})),
    };
    const std::vector<int32_t> fshape = {1, 2, 2, 3};
    const std::vector<int32_t> rshape = {1};
    const std::vector<flatbuffers::Offset<Tensor>> tensors = {
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                     TensorType_RESOURCE),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                     TensorType_FLOAT32),
    };
    const auto vh = CreateOperator(
        builder, 0, builder.CreateVector<int32_t>({}),
        builder.CreateVector<int32_t>({0}),
        tflite::BuiltinOptions_VarHandleOptions,
        CreateVarHandleOptions(builder, builder.CreateString("c"),
                               builder.CreateString("ro_v"))
            .Union());
    const auto rv =
        CreateOperator(builder, 1, builder.CreateVector<int32_t>({0}),
                       builder.CreateVector<int32_t>({1}));
    const auto sg = CreateSubGraph(
        builder, builder.CreateVector(tensors.data(), tensors.size()),
        builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({1}),
        builder.CreateVector({vh, rv}));
    builder.Finish(CreateModel(
        builder, TFLITE_SCHEMA_VERSION,
        builder.CreateVector(opcodes.data(), opcodes.size()),
        builder.CreateVector(&sg, 1), builder.CreateString("ReadOnlyModel"),
        builder.CreateVector(buffers.data(), buffers.size())));
    const std::vector<char> buf(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        reinterpret_cast<const char*>(builder.GetBufferPointer()) +
            builder.GetSize());
    return make_interp_from_buf(buf, backing);
  };

  auto fresh_delegate = []() {
    auto d = NewXnnPackDelegateSupportingVariableOps();
    d->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
    return d;
  };

  struct SpyCapture {
    std::vector<int> replaced_nodes;
    TfLiteRegistration subgraph_reg{};
    bool saw_replace_call = false;
    int report_error_count = 0;
  };
  static thread_local SpyCapture* g_active_spy = nullptr;
  struct SpyCallbacks {
    static void ReportError(TfLiteContext*, const char*, ...) {
      if (g_active_spy != nullptr) {
        ++g_active_spy->report_error_count;
      }
    }
  };

  TfLiteDelegate probe = TfLiteDelegateCreate();
  probe.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  auto run_in_delegate_ctx =
      [&probe](Interpreter* interp,
               const std::function<void(TfLiteContext*)>& fn) {
        probe.data_ = const_cast<std::function<void(TfLiteContext*)>*>(&fn);
        probe.Prepare = [](TfLiteContext* real_ctx,
                           TfLiteDelegate* self) -> TfLiteStatus {
          auto* cb =
              static_cast<std::function<void(TfLiteContext*)>*>(self->data_);
          (*cb)(real_ctx);
          return kTfLiteOk;
        };
        ASSERT_EQ(interp->ModifyGraphWithDelegate(&probe), kTfLiteOk);
      };

  auto make_spy_ctx = [](TfLiteContext* real_ctx, SpyCapture* cap) {
    cap->replaced_nodes.clear();
    cap->saw_replace_call = false;
    cap->report_error_count = 0;
    g_active_spy = cap;
    TfLiteContext spy_ctx = *real_ctx;
    spy_ctx.ReportError = &SpyCallbacks::ReportError;
    spy_ctx.ReplaceNodeSubsetsWithDelegateKernels =
        [](TfLiteContext*, TfLiteRegistration reg,
           const TfLiteIntArray* nodes_to_replace,
           TfLiteDelegate*) -> TfLiteStatus {
      if (g_active_spy != nullptr) {
        g_active_spy->saw_replace_call = true;
        g_active_spy->subgraph_reg = reg;
        if (nodes_to_replace != nullptr && nodes_to_replace->size > 0) {
          g_active_spy->replaced_nodes.assign(
              nodes_to_replace->data,
              nodes_to_replace->data + nodes_to_replace->size);
        }
      }
      return kTfLiteOk;
    };
    return spy_ctx;
  };

  auto make_int_array = [](const std::vector<int>& vals) {
    TfLiteIntArray* arr = TfLiteIntArrayCreate(static_cast<int>(vals.size()));
    for (size_t i = 0; i < vals.size(); ++i) {
      arr->data[i] = vals[i];
    }
    return arr;
  };

  // Capture kSubgraphRegistration from a clean positive control interpreter.
  TfLiteRegistration k_subgraph_reg{};
  {
    std::vector<char> ctrl_backing;
    auto ctrl_interp = make_assign_then_read_interp(&ctrl_backing);
    ASSERT_TRUE(ctrl_interp);
    run_in_delegate_ctx(ctrl_interp.get(), [&](TfLiteContext* real_ctx) {
      SpyCapture cap;
      TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
      auto d = fresh_delegate();
      ASSERT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
      ASSERT_TRUE(cap.saw_replace_call);
      ASSERT_EQ(cap.replaced_nodes, (std::vector<int>{0, 1, 2}));
      k_subgraph_reg = cap.subgraph_reg;
      ASSERT_NE(k_subgraph_reg.init, nullptr);
      ASSERT_NE(k_subgraph_reg.free, nullptr);
    });
  }

  static TfLiteNode* g_synth_node = nullptr;
  static TfLiteRegistration* g_synth_reg = nullptr;

  auto expect_forced_subgraph_create_rejected =
      [&](TfLiteContext* spy_ctx, TfLiteDelegate* d,
          const std::vector<int>& node_ids, const std::vector<int>& in_ids,
          const std::vector<int>& out_ids) {
        TfLiteIntArray* nodes = make_int_array(node_ids);
        TfLiteIntArray* ins = make_int_array(in_ids);
        TfLiteIntArray* outs = make_int_array(out_ids);
        TfLiteDelegateParams params{};
        params.delegate = d;
        params.nodes_to_replace = nodes;
        params.input_tensors = ins;
        params.output_tensors = outs;
        void* sg = k_subgraph_reg.init(
            spy_ctx, reinterpret_cast<const char*>(&params), sizeof(params));
        EXPECT_EQ(sg, nullptr);
        if (sg != nullptr) {
          k_subgraph_reg.free(spy_ctx, sg);
        }
        TfLiteIntArrayFree(nodes);
        TfLiteIntArrayFree(ins);
        TfLiteIntArrayFree(outs);
      };

  auto run_synthetic_subgraph_create_op_case =
      [&](int32_t builtin_code, const std::vector<int>& raw_inputs,
          const std::vector<int>& raw_outputs, int first_int_param = 0,
          const char* custom_name = nullptr, bool null_builtin_data = false,
          int custom_initial_data_size_override = -9999) {
        std::vector<char> backing;
        auto interp = make_assign_then_read_interp(&backing);
        ASSERT_TRUE(interp);
        run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
          std::vector<int> resolved_inputs = raw_inputs;
          for (int& v : resolved_inputs) {
            if (v == 999) {
              v = static_cast<int>(real_ctx->tensors_size);
            }
          }
          alignas(16) int builtin_params[16] = {};
          builtin_params[0] = first_int_param;

          static const int32_t kUnitStrides4[4] = {1, 1, 1, 1};
          const bool setup_static_stride_t1 =
              (builtin_code == BuiltinOperator_STRIDED_SLICE &&
               raw_outputs.empty());
          TfLiteTensor saved_t1{};
          TfLiteIntArray* stride_dims = nullptr;
          if (setup_static_stride_t1) {
            saved_t1 = real_ctx->tensors[1];
            stride_dims = make_int_array({4});
            real_ctx->tensors[1].type = kTfLiteInt32;
            real_ctx->tensors[1].allocation_type = kTfLiteMmapRo;
            real_ctx->tensors[1].dims = stride_dims;
            real_ctx->tensors[1].data.i32 =
                const_cast<int32_t*>(kUnitStrides4);
            real_ctx->tensors[1].bytes = sizeof(kUnitStrides4);
          }

          flexbuffers::Builder sdpa_attr_builder;
          if (custom_name != nullptr) {
            sdpa_attr_builder.Map(
                [&]() { sdpa_attr_builder.Float("scale", 0.5f); });
            sdpa_attr_builder.Finish();
          }

          TfLiteNode synth_node{};
          synth_node.inputs = make_int_array(resolved_inputs);
          synth_node.outputs = make_int_array(raw_outputs);
          synth_node.builtin_data =
              null_builtin_data ? nullptr : static_cast<void*>(builtin_params);
          if (custom_name != nullptr) {
            synth_node.custom_initial_data =
                sdpa_attr_builder.GetBuffer().data();
            synth_node.custom_initial_data_size =
                (custom_initial_data_size_override != -9999)
                    ? custom_initial_data_size_override
                    : static_cast<int>(sdpa_attr_builder.GetSize());
          }

          TfLiteRegistration synth_reg{};
          synth_reg.builtin_code = builtin_code;
          synth_reg.custom_name = custom_name;

          g_synth_node = &synth_node;
          g_synth_reg = &synth_reg;

          SpyCapture cap;
          TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
          spy_ctx.GetNodeAndRegistration =
              [](TfLiteContext*, int node_index, TfLiteNode** node,
                 TfLiteRegistration** registration) -> TfLiteStatus {
            if (node_index == 0 && g_synth_node != nullptr &&
                g_synth_reg != nullptr) {
              *node = g_synth_node;
              *registration = g_synth_reg;
              return kTfLiteOk;
            }
            return kTfLiteError;
          };

          auto d = fresh_delegate();
          expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0}, {0},
                                                 {2});
          if (null_builtin_data ||
              custom_initial_data_size_override != -9999) {
            static TfLiteIntArray* kOneNodePlan = nullptr;
            TfLiteIntArray* plan = make_int_array({0});
            kOneNodePlan = plan;
            SpyCapture prep_cap;
            TfLiteContext prep_ctx = make_spy_ctx(real_ctx, &prep_cap);
            prep_ctx.GetNodeAndRegistration = spy_ctx.GetNodeAndRegistration;
            prep_ctx.GetExecutionPlan = [](TfLiteContext*,
                                           TfLiteIntArray** out_plan) {
              *out_plan = kOneNodePlan;
              return kTfLiteOk;
            };
            auto d_prep = fresh_delegate();
            EXPECT_EQ(d_prep->Prepare(&prep_ctx, d_prep.get()), kTfLiteOk);
            EXPECT_TRUE(prep_cap.replaced_nodes.empty());
            kOneNodePlan = nullptr;
            TfLiteIntArrayFree(plan);
          }

          g_synth_node = nullptr;
          g_synth_reg = nullptr;
          if (setup_static_stride_t1) {
            real_ctx->tensors[1] = saved_t1;
            TfLiteIntArrayFree(stride_dims);
          }
          TfLiteIntArrayFree(synth_node.inputs);
          TfLiteIntArrayFree(synth_node.outputs);
        });
      };

  auto run_null_node_array_case = [&](int target_node, bool null_inputs) {
    std::vector<char> backing;
    auto interp = make_assign_then_read_interp(&backing);
    ASSERT_TRUE(interp);
    run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
      const auto* nr = interp->node_and_registration(target_node);
      ASSERT_NE(nr, nullptr);
      TfLiteNode* mut_node = const_cast<TfLiteNode*>(&nr->first);
      TfLiteIntArray*& field =
          null_inputs ? mut_node->inputs : mut_node->outputs;
      TfLiteIntArray* saved = field;
      field = nullptr;

      SpyCapture cap;
      TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
      auto d = fresh_delegate();
      EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
      EXPECT_EQ(std::find(cap.replaced_nodes.begin(), cap.replaced_nodes.end(),
                          target_node),
                cap.replaced_nodes.end());
      expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1, 2}, {0},
                                             {2});
      field = saved;
    });
  };

  auto run_index_mutation_case = [&](int node_idx, bool is_output, int slot,
                                     int bad_index,
                                     const std::vector<int>& create_nodes) {
    std::vector<char> backing;
    auto interp = make_assign_then_read_interp(&backing);
    ASSERT_TRUE(interp);
    run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
      const int actual_bad =
          (bad_index == 999) ? static_cast<int>(interp->tensors_size())
                             : bad_index;
      const auto* nr = interp->node_and_registration(node_idx);
      ASSERT_NE(nr, nullptr);
      TfLiteIntArray* arr = is_output ? nr->first.outputs : nr->first.inputs;
      const int saved = arr->data[slot];
      arr->data[slot] = actual_bad;

      SpyCapture cap;
      TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
      auto d = fresh_delegate();
      EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
      EXPECT_EQ(std::find(cap.replaced_nodes.begin(), cap.replaced_nodes.end(),
                          node_idx),
                cap.replaced_nodes.end());
      expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), create_nodes,
                                             {0}, {2});
      arr->data[slot] = saved;
    });
  };

  auto run_assign_shape_case = [&](bool set_null,
                                   const std::vector<int>& bad_shape) {
    std::vector<char> backing;
    auto interp = make_assign_then_read_interp(&backing);
    ASSERT_TRUE(interp);
    run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
      TfLiteTensor* t = interp->tensor(0);
      ASSERT_NE(t, nullptr);
      ASSERT_EQ(t->type, kTfLiteFloat32);
      TfLiteIntArray* saved_dims = t->dims;
      TfLiteIntArray* bad_dims = set_null ? nullptr : make_int_array(bad_shape);
      t->dims = bad_dims;

      SpyCapture cap;
      TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
      auto d = fresh_delegate();
      EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
      EXPECT_EQ(std::find(cap.replaced_nodes.begin(), cap.replaced_nodes.end(),
                          1),
                cap.replaced_nodes.end());
      expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1, 2}, {0},
                                             {2});
      t->dims = saved_dims;
      if (bad_dims != nullptr) {
        TfLiteIntArrayFree(bad_dims);
      }
    });
  };

  auto run_read_only_shape_case = [&](bool set_null,
                                      const std::vector<int>& bad_shape) {
    std::vector<char> backing;
    auto interp = make_read_only_interp(&backing);
    ASSERT_TRUE(interp);
    run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
      TfLiteTensor* t = interp->tensor(1);
      ASSERT_NE(t, nullptr);
      ASSERT_EQ(t->type, kTfLiteFloat32);
      TfLiteIntArray* saved_dims = t->dims;
      TfLiteIntArray* bad_dims = set_null ? nullptr : make_int_array(bad_shape);
      t->dims = bad_dims;

      SpyCapture cap;
      TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
      auto d = fresh_delegate();
      EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
      EXPECT_TRUE(cap.replaced_nodes.empty());
      expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1}, {},
                                             {1});
      t->dims = saved_dims;
      if (bad_dims != nullptr) {
        TfLiteIntArrayFree(bad_dims);
      }
    });
  };

  switch (test_case) {
    case VariableOpGuardCase::kNullContextInPrepare: {
      auto d = fresh_delegate();
      EXPECT_EQ(d->Prepare(nullptr, d.get()), kTfLiteError);
      break;
    }
    case VariableOpGuardCase::kNullExecutionPlanInPrepare: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        SpyCapture cap;
        TfLiteContext bad_ctx = make_spy_ctx(real_ctx, &cap);
        bad_ctx.GetExecutionPlan = [](TfLiteContext*, TfLiteIntArray** plan) {
          *plan = nullptr;
          return kTfLiteOk;
        };
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&bad_ctx, d.get()), kTfLiteError);
      });
      break;
    }
    case VariableOpGuardCase::kNullParamsInSubgraphCreate: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        void* sg = k_subgraph_reg.init(&spy_ctx, nullptr, 0);
        EXPECT_EQ(sg, nullptr);
        if (sg != nullptr) {
          k_subgraph_reg.free(&spy_ctx, sg);
        }
      });
      break;
    }
    case VariableOpGuardCase::kNegativeInputTensorsSizeInSubgraphCreate: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        ASSERT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        TfLiteIntArray* nodes = make_int_array({0, 1, 2});
        TfLiteIntArray* ins = make_int_array({0});
        TfLiteIntArray* outs = make_int_array({2});
        ins->size = -1;
        TfLiteDelegateParams params{};
        params.delegate = d.get();
        params.nodes_to_replace = nodes;
        params.input_tensors = ins;
        params.output_tensors = outs;
        void* sg = k_subgraph_reg.init(
            &spy_ctx, reinterpret_cast<const char*>(&params), sizeof(params));
        EXPECT_EQ(sg, nullptr);
        if (sg != nullptr) {
          k_subgraph_reg.free(&spy_ctx, sg);
        }
        ins->size = 1;
        TfLiteIntArrayFree(nodes);
        TfLiteIntArrayFree(ins);
        TfLiteIntArrayFree(outs);
      });
      break;
    }
    case VariableOpGuardCase::kOutOfBoundsOutputTensorIndexInSubgraphCreate: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        ASSERT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        expect_forced_subgraph_create_rejected(
            &spy_ctx, d.get(), {0, 1, 2}, {0},
            {static_cast<int>(interp->tensors_size())});
      });
      break;
    }
    case VariableOpGuardCase::kNullNodeInputsVarHandle:
      run_null_node_array_case(0, /*null_inputs=*/true);
      break;
    case VariableOpGuardCase::kNullNodeInputsAssignVariable:
      run_null_node_array_case(1, /*null_inputs=*/true);
      break;
    case VariableOpGuardCase::kNullNodeInputsReadVariable:
      run_null_node_array_case(2, /*null_inputs=*/true);
      break;
    case VariableOpGuardCase::kNullNodeOutputsVarHandle:
      run_null_node_array_case(0, /*null_inputs=*/false);
      break;
    case VariableOpGuardCase::kNullNodeOutputsAssignVariable:
      run_null_node_array_case(1, /*null_inputs=*/false);
      break;
    case VariableOpGuardCase::kNullNodeOutputsReadVariable:
      run_null_node_array_case(2, /*null_inputs=*/false);
      break;
    case VariableOpGuardCase::kNegativeInputsSizeVarHandle: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        const auto* nr = interp->node_and_registration(0);
        ASSERT_NE(nr, nullptr);
        TfLiteNode* mut_node = const_cast<TfLiteNode*>(&nr->first);
        const int saved_size = mut_node->inputs->size;
        mut_node->inputs->size = -1;

        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        EXPECT_TRUE(cap.replaced_nodes.empty());
        expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0}, {}, {});
        mut_node->inputs->size = saved_size;
      });
      break;
    }
    case VariableOpGuardCase::kUnexpectedInputArityVarHandle: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        const auto* nr = interp->node_and_registration(0);
        ASSERT_NE(nr, nullptr);
        TfLiteNode* mut_node = const_cast<TfLiteNode*>(&nr->first);
        TfLiteIntArray* saved_inputs = mut_node->inputs;
        TfLiteIntArray* extra_inputs = make_int_array({0});
        mut_node->inputs = extra_inputs;

        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        EXPECT_TRUE(cap.replaced_nodes.empty());
        expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0}, {}, {});
        mut_node->inputs = saved_inputs;
        TfLiteIntArrayFree(extra_inputs);
      });
      break;
    }
    case VariableOpGuardCase::kUnexpectedOutputArityAssignVariable: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        const auto* nr = interp->node_and_registration(1);
        ASSERT_NE(nr, nullptr);
        TfLiteNode* mut_node = const_cast<TfLiteNode*>(&nr->first);
        TfLiteIntArray* saved_outputs = mut_node->outputs;
        TfLiteIntArray* extra_outputs = make_int_array({2});
        mut_node->outputs = extra_outputs;

        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        EXPECT_EQ(std::find(cap.replaced_nodes.begin(),
                            cap.replaced_nodes.end(), 1),
                  cap.replaced_nodes.end());
        expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1, 2},
                                               {0}, {2});
        mut_node->outputs = saved_outputs;
        TfLiteIntArrayFree(extra_outputs);
      });
      break;
    }
    case VariableOpGuardCase::kNullVarHandleBuiltinData: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        const auto* nr = interp->node_and_registration(0);
        ASSERT_NE(nr, nullptr);
        TfLiteNode* mut_node = const_cast<TfLiteNode*>(&nr->first);
        void* saved_params = mut_node->builtin_data;
        mut_node->builtin_data = nullptr;

        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        EXPECT_TRUE(cap.replaced_nodes.empty());
        expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1, 2},
                                               {0}, {2});
        mut_node->builtin_data = saved_params;
      });
      break;
    }
    case VariableOpGuardCase::kVarHandleOutputNegativeOne:
      run_index_mutation_case(0, /*is_output=*/true, 0, -1, {0});
      break;
    case VariableOpGuardCase::kVarHandleOutputNegativeTwo:
      run_index_mutation_case(0, /*is_output=*/true, 0, -2, {0});
      break;
    case VariableOpGuardCase::kVarHandleOutputOutOfBounds:
      run_index_mutation_case(0, /*is_output=*/true, 0, 999, {0});
      break;
    case VariableOpGuardCase::kAssignResourceInputNegativeOne:
      run_index_mutation_case(1, /*is_output=*/false, 0, -1, {0, 1, 2});
      break;
    case VariableOpGuardCase::kAssignResourceInputNegativeTwo:
      run_index_mutation_case(1, /*is_output=*/false, 0, -2, {0, 1, 2});
      break;
    case VariableOpGuardCase::kAssignResourceInputOutOfBounds:
      run_index_mutation_case(1, /*is_output=*/false, 0, 999, {0, 1, 2});
      break;
    case VariableOpGuardCase::kAssignValueInputNegativeOne:
      run_index_mutation_case(1, /*is_output=*/false, 1, -1, {0, 1, 2});
      break;
    case VariableOpGuardCase::kAssignValueInputNegativeTwo:
      run_index_mutation_case(1, /*is_output=*/false, 1, -2, {0, 1, 2});
      break;
    case VariableOpGuardCase::kAssignValueInputOutOfBounds:
      run_index_mutation_case(1, /*is_output=*/false, 1, 999, {0, 1, 2});
      break;
    case VariableOpGuardCase::kReadResourceInputNegativeOne:
      run_index_mutation_case(2, /*is_output=*/false, 0, -1, {0, 1, 2});
      break;
    case VariableOpGuardCase::kReadResourceInputNegativeTwo:
      run_index_mutation_case(2, /*is_output=*/false, 0, -2, {0, 1, 2});
      break;
    case VariableOpGuardCase::kReadResourceInputOutOfBounds:
      run_index_mutation_case(2, /*is_output=*/false, 0, 999, {0, 1, 2});
      break;
    case VariableOpGuardCase::kReadValueOutputNegativeOne:
      run_index_mutation_case(2, /*is_output=*/true, 0, -1, {0, 1, 2});
      break;
    case VariableOpGuardCase::kReadValueOutputNegativeTwo:
      run_index_mutation_case(2, /*is_output=*/true, 0, -2, {0, 1, 2});
      break;
    case VariableOpGuardCase::kReadValueOutputOutOfBounds:
      run_index_mutation_case(2, /*is_output=*/true, 0, 999, {0, 1, 2});
      break;
    case VariableOpGuardCase::kWrongResourceTensorType: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        TfLiteTensor* res_t = interp->tensor(1);
        ASSERT_NE(res_t, nullptr);
        ASSERT_EQ(res_t->type, kTfLiteResource);
        res_t->type = kTfLiteFloat32;

        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        EXPECT_TRUE(cap.replaced_nodes.empty());
        expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1, 2},
                                               {0}, {2});
        res_t->type = kTfLiteResource;
      });
      break;
    }
    case VariableOpGuardCase::kAssignInputNullDims:
      run_assign_shape_case(/*set_null=*/true, {});
      break;
    case VariableOpGuardCase::kAssignInputScalarRankZero:
      run_assign_shape_case(/*set_null=*/false, {});
      break;
    case VariableOpGuardCase::kAssignInputNegativeDim:
      run_assign_shape_case(/*set_null=*/false, {-1, 2, 2, 3});
      break;
    case VariableOpGuardCase::kAssignInputExcessiveRank:
      run_assign_shape_case(/*set_null=*/false, {1, 1, 1, 1, 1, 1, 1, 2});
      break;
    case VariableOpGuardCase::kAssignInputByteSizeOverflow:
      run_assign_shape_case(/*set_null=*/false, {65536, 65536, 65536, 32768});
      break;
    case VariableOpGuardCase::kAssignInputShapeExceedsTensorBytes: {
      // Mutate ASSIGN_VARIABLE input dims from {1, 2, 2, 3} (48B) to
      // non-overflowing {1, 5, 2, 3} (120B) while backing buffer remains 48B.
      std::vector<char> backing;
      auto interp = make_assign_only_interp(&backing);
      ASSERT_TRUE(interp);
      const int in_idx = interp->inputs()[0];
      TfLiteTensor* t0 = interp->tensor(in_idx);
      ASSERT_NE(t0, nullptr);
      ASSERT_EQ(t0->bytes, 48u);

      char* orig_raw = t0->data.raw;
      const size_t orig_bytes = t0->bytes;
      TfLiteIntArray* orig_dims = t0->dims;

      char* exact_48b = static_cast<char*>(std::calloc(48, 1));
      ASSERT_NE(exact_48b, nullptr);
      TfLiteIntArray* inflated_dims = make_int_array({1, 5, 2, 3});
      t0->data.raw = exact_48b;
      t0->bytes = 48;
      t0->dims = inflated_dims;

      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d_spy = fresh_delegate();
        EXPECT_EQ(d_spy->Prepare(&spy_ctx, d_spy.get()), kTfLiteOk);
        EXPECT_TRUE(cap.replaced_nodes.empty());
        expect_forced_subgraph_create_rejected(&spy_ctx, d_spy.get(), {0, 1},
                                               {in_idx}, {});
      });

      auto d_live = fresh_delegate();
      ASSERT_EQ(interp->ModifyGraphWithDelegate(d_live.get()), kTfLiteOk);
      EXPECT_EQ(interp->Invoke(), kTfLiteOk);

      t0->data.raw = orig_raw;
      t0->bytes = orig_bytes;
      t0->dims = orig_dims;
      TfLiteIntArrayFree(inflated_dims);
      std::free(exact_48b);
      break;
    }
    case VariableOpGuardCase::kReadOutputNullDims:
      run_read_only_shape_case(/*set_null=*/true, {});
      break;
    case VariableOpGuardCase::kReadOutputNegativeDim:
      run_read_only_shape_case(/*set_null=*/false, {-1, 2, 2, 3});
      break;
    case VariableOpGuardCase::kReadOutputExcessiveRank:
      run_read_only_shape_case(/*set_null=*/false, {1, 1, 1, 1, 1, 1, 1, 2});
      break;
    case VariableOpGuardCase::kReadOutputByteSizeOverflow:
      run_read_only_shape_case(/*set_null=*/false,
                               {65536, 65536, 65536, 32768});
      break;
    case VariableOpGuardCase::kReadOutputShapeExceedsTensorBytes: {
      // Secondary CALL_ONCE subgraph initializes ("c", "v") to {1, 5, 2, 3}
      // (120B), while primary READ_VARIABLE output tensor has a 48B allocation
      // with dims mutated to {1, 5, 2, 3} (120B > 48B).
      flatbuffers::FlatBufferBuilder builder;
      const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
          CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
          CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
          CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
          CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
      };
      const std::vector<float> init_vals(30, 2.5f);
      const std::vector<flatbuffers::Offset<Buffer>> buffers = {
          CreateBuffer(builder, builder.CreateVector({})),
          CreateBuffer(
              builder,
              builder.CreateVector(
                  reinterpret_cast<const uint8_t*>(init_vals.data()),
                  init_vals.size() * sizeof(float))),
      };
      const std::vector<int32_t> small_shape = {1, 2, 2, 3};
      const std::vector<int32_t> large_shape = {1, 5, 2, 3};
      const std::vector<int32_t> rshape = {1};

      const std::vector<flatbuffers::Offset<Tensor>> primary_tensors = {
          CreateTensor(
              builder,
              builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
              TensorType_RESOURCE),
          CreateTensor(builder,
                       builder.CreateVector<int32_t>(small_shape.data(),
                                                     small_shape.size()),
                       TensorType_FLOAT32),
      };
      const auto co = CreateOperator(
          builder, 3, builder.CreateVector<int32_t>({}),
          builder.CreateVector<int32_t>({}),
          tflite::BuiltinOptions_CallOnceOptions,
          CreateCallOnceOptions(builder, 1).Union());
      const auto p_vh = CreateOperator(
          builder, 0, builder.CreateVector<int32_t>({}),
          builder.CreateVector<int32_t>({0}),
          tflite::BuiltinOptions_VarHandleOptions,
          CreateVarHandleOptions(builder, builder.CreateString("c"),
                                 builder.CreateString("v"))
              .Union());
      const auto p_rv =
          CreateOperator(builder, 1, builder.CreateVector<int32_t>({0}),
                         builder.CreateVector<int32_t>({1}));
      const auto primary_sg = CreateSubGraph(
          builder,
          builder.CreateVector(primary_tensors.data(), primary_tensors.size()),
          builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({1}),
          builder.CreateVector({co, p_vh, p_rv}));

      const std::vector<flatbuffers::Offset<Tensor>> secondary_tensors = {
          CreateTensor(
              builder,
              builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
              TensorType_RESOURCE),
          CreateTensor(builder,
                       builder.CreateVector<int32_t>(large_shape.data(),
                                                     large_shape.size()),
                       TensorType_FLOAT32, /*buffer=*/1),
      };
      const auto s_vh = CreateOperator(
          builder, 0, builder.CreateVector<int32_t>({}),
          builder.CreateVector<int32_t>({0}),
          tflite::BuiltinOptions_VarHandleOptions,
          CreateVarHandleOptions(builder, builder.CreateString("c"),
                                 builder.CreateString("v"))
              .Union());
      const auto s_av =
          CreateOperator(builder, 2, builder.CreateVector<int32_t>({0, 1}),
                         builder.CreateVector<int32_t>({}));
      const auto secondary_sg = CreateSubGraph(
          builder,
          builder.CreateVector(secondary_tensors.data(),
                               secondary_tensors.size()),
          builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({}),
          builder.CreateVector({s_vh, s_av}));

      builder.Finish(CreateModel(
          builder, TFLITE_SCHEMA_VERSION,
          builder.CreateVector(opcodes.data(), opcodes.size()),
          builder.CreateVector({primary_sg, secondary_sg}),
          builder.CreateString("ReadOutputExceedsBytesModel"),
          builder.CreateVector(buffers.data(), buffers.size())));

      const Model* model = GetModel(builder.GetBufferPointer());
      std::unique_ptr<Interpreter> interp;
      ASSERT_EQ(
          InterpreterBuilder(
              model, ::tflite::ops::builtin::
                         BuiltinOpResolverWithoutDefaultDelegates())(&interp),
          kTfLiteOk);
      ASSERT_TRUE(interp);
      ASSERT_EQ(interp->AllocateTensors(), kTfLiteOk);

      TfLiteTensor* t1 = interp->tensor(1);
      ASSERT_NE(t1, nullptr);
      ASSERT_EQ(t1->bytes, 48u);
      char* orig_raw = t1->data.raw;
      const size_t orig_bytes = t1->bytes;
      TfLiteIntArray* orig_dims = t1->dims;

      char* exact_48b = static_cast<char*>(std::calloc(48, 1));
      ASSERT_NE(exact_48b, nullptr);
      TfLiteIntArray* inflated_dims = make_int_array({1, 5, 2, 3});
      t1->data.raw = exact_48b;
      t1->bytes = 48;
      t1->dims = inflated_dims;

      auto d_live = fresh_delegate();
      ASSERT_EQ(interp->ModifyGraphWithDelegate(d_live.get()), kTfLiteOk);
      EXPECT_EQ(interp->Invoke(), kTfLiteOk);

      t1->data.raw = orig_raw;
      t1->bytes = orig_bytes;
      t1->dims = orig_dims;
      TfLiteIntArrayFree(inflated_dims);
      std::free(exact_48b);
      break;
    }
    case VariableOpGuardCase::kConflictingVarHandlesOnSameLocalResourceTensor: {
      flatbuffers::FlatBufferBuilder builder;
      const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
          CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
          CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
          CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      };
      const std::vector<flatbuffers::Offset<Buffer>> buffers = {
          CreateBuffer(builder, builder.CreateVector({})),
      };
      const std::vector<int32_t> fshape = {1, 2, 2, 3};
      const std::vector<int32_t> rshape = {1};
      const std::vector<flatbuffers::Offset<Tensor>> tensors = {
          CreateTensor(
              builder,
              builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
              TensorType_RESOURCE),
          CreateTensor(
              builder,
              builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
              TensorType_FLOAT32),
          CreateTensor(
              builder,
              builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
              TensorType_FLOAT32),
      };
      const auto vh1 = CreateOperator(
          builder, 0, builder.CreateVector<int32_t>({}),
          builder.CreateVector<int32_t>({0}),
          tflite::BuiltinOptions_VarHandleOptions,
          CreateVarHandleOptions(builder, builder.CreateString("c"),
                                 builder.CreateString("v_one"))
              .Union());
      const auto vh2 = CreateOperator(
          builder, 0, builder.CreateVector<int32_t>({}),
          builder.CreateVector<int32_t>({0}),
          tflite::BuiltinOptions_VarHandleOptions,
          CreateVarHandleOptions(builder, builder.CreateString("c"),
                                 builder.CreateString("v_two"))
              .Union());
      const auto av =
          CreateOperator(builder, 2, builder.CreateVector<int32_t>({0, 1}),
                         builder.CreateVector<int32_t>({}));
      const auto rv =
          CreateOperator(builder, 1, builder.CreateVector<int32_t>({0}),
                         builder.CreateVector<int32_t>({2}));
      const auto sg = CreateSubGraph(
          builder, builder.CreateVector(tensors.data(), tensors.size()),
          builder.CreateVector<int32_t>({1}),
          builder.CreateVector<int32_t>({2}),
          builder.CreateVector({vh1, vh2, av, rv}));
      builder.Finish(CreateModel(
          builder, TFLITE_SCHEMA_VERSION,
          builder.CreateVector(opcodes.data(), opcodes.size()),
          builder.CreateVector(&sg, 1), builder.CreateString("ConflictModel"),
          builder.CreateVector(buffers.data(), buffers.size())));

      const Model* model = GetModel(builder.GetBufferPointer());
      std::unique_ptr<Interpreter> interp;
      ASSERT_EQ(
          InterpreterBuilder(
              model, ::tflite::ops::builtin::
                         BuiltinOpResolverWithoutDefaultDelegates())(&interp),
          kTfLiteOk);
      ASSERT_TRUE(interp);
      ASSERT_EQ(interp->AllocateTensors(), kTfLiteOk);
      run_in_delegate_ctx(interp.get(), [&](TfLiteContext* real_ctx) {
        SpyCapture cap;
        TfLiteContext spy_ctx = make_spy_ctx(real_ctx, &cap);
        auto d = fresh_delegate();
        EXPECT_EQ(d->Prepare(&spy_ctx, d.get()), kTfLiteOk);
        EXPECT_TRUE(cap.replaced_nodes.empty());
        expect_forced_subgraph_create_rejected(&spy_ctx, d.get(), {0, 1, 2, 3},
                                               {1}, {2});
      });
      break;
    }
    case VariableOpGuardCase::
        kExternalOutputBytesSmallerThanXnnpackShapeAtInvoke: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      auto d = fresh_delegate();
      ASSERT_EQ(interp->ModifyGraphWithDelegate(d.get()), kTfLiteOk);

      const int out_idx = interp->outputs()[0];
      TfLiteTensor* out_t = interp->tensor(out_idx);
      ASSERT_NE(out_t, nullptr);
      ASSERT_EQ(out_t->bytes, 48u);

      char* orig_raw = out_t->data.raw;
      const size_t orig_bytes = out_t->bytes;
      char* shrunk_16b = static_cast<char*>(std::calloc(16, 1));
      ASSERT_NE(shrunk_16b, nullptr);
      out_t->data.raw = shrunk_16b;
      out_t->bytes = 16;

      EXPECT_EQ(interp->Invoke(), kTfLiteError);

      out_t->data.raw = orig_raw;
      out_t->bytes = orig_bytes;
      std::free(shrunk_16b);
      break;
    }
    case VariableOpGuardCase::
        kExternalNullDataWithNonZeroXnnpackShapeAtInvoke: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      auto d = fresh_delegate();
      ASSERT_EQ(interp->ModifyGraphWithDelegate(d.get()), kTfLiteOk);

      const int out_idx = interp->outputs()[0];
      TfLiteTensor* out_t = interp->tensor(out_idx);
      ASSERT_NE(out_t, nullptr);
      ASSERT_EQ(out_t->bytes, 48u);

      char* orig_raw = out_t->data.raw;
      const size_t orig_bytes = out_t->bytes;
      out_t->data.raw = nullptr;
      out_t->bytes = 0;

      EXPECT_EQ(interp->Invoke(), kTfLiteError);

      out_t->data.raw = orig_raw;
      out_t->bytes = orig_bytes;
      break;
    }
    case VariableOpGuardCase::
        kReshapedExternalInputBytesSmallerThanShapeInPrepare: {
      std::vector<char> backing;
      auto interp = make_assign_then_read_interp(&backing);
      ASSERT_TRUE(interp);
      auto d = fresh_delegate();
      ASSERT_EQ(interp->ModifyGraphWithDelegate(d.get()), kTfLiteOk);

      const int in_idx = interp->inputs()[0];
      ASSERT_EQ(interp->ResizeInputTensor(in_idx, {1, 5, 2, 3}), kTfLiteOk);
      TfLiteTensor* t0 = interp->tensor(in_idx);
      ASSERT_NE(t0, nullptr);
      const TfLiteAllocationType orig_alloc_type = t0->allocation_type;
      char* orig_raw = t0->data.raw;
      const size_t orig_bytes = t0->bytes;

      char* exact_48b = static_cast<char*>(std::calloc(48, 1));
      ASSERT_NE(exact_48b, nullptr);
      t0->allocation_type = kTfLiteCustom;
      t0->data.raw = exact_48b;
      t0->bytes = 48;

      const TfLiteStatus alloc_status = interp->AllocateTensors();
      EXPECT_EQ(alloc_status, kTfLiteError);
      if (alloc_status == kTfLiteOk) {
        EXPECT_EQ(interp->Invoke(), kTfLiteError);
      }

      t0->allocation_type = orig_alloc_type;
      t0->data.raw = orig_raw;
      t0->bytes = orig_bytes;
      std::free(exact_48b);
      break;
    }
    case VariableOpGuardCase::kCreateReshapeZeroInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_RESHAPE, {}, {2});
      break;
    case VariableOpGuardCase::kCreateReshapeOutOfBoundsInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_RESHAPE, {999},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateReshapeOutOfBoundsShapeInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_RESHAPE, {0, 131},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateMeanZeroInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_MEAN, {}, {2});
      break;
    case VariableOpGuardCase::kCreatePadOutOfBoundsPaddingsInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_PAD, {0, 131}, {2});
      break;
    case VariableOpGuardCase::kCreateResizeBilinearZeroInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_RESIZE_BILINEAR, {},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateSliceLackingInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_SLICE, {0}, {2});
      break;
    case VariableOpGuardCase::kCreateStridedSliceLackingInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_STRIDED_SLICE, {0},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateSplitMissingInputSlot1:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_SPLIT, {0}, {2},
                                            /*first_int_param=*/2);
      break;
    case VariableOpGuardCase::kCreateSplitOutOfBoundsValueInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_SPLIT, {0, 999},
                                            {2}, /*first_int_param=*/2);
      break;
    case VariableOpGuardCase::kCreateSplitOutOfBoundsAxisInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_SPLIT, {131, 0},
                                            {0, 2}, /*first_int_param=*/2);
      break;
    case VariableOpGuardCase::kCreateTransposeZeroInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_TRANSPOSE, {}, {2});
      break;
    case VariableOpGuardCase::kCreateTransposeOutOfBoundsInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_TRANSPOSE, {999, 0},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateTransposeOutOfBoundsPermInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_TRANSPOSE, {0, 131},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateTransposeConvOutOfBoundsFilterInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_TRANSPOSE_CONV,
                                            {0, 999, 0}, {2});
      break;
    case VariableOpGuardCase::kCreateTransposeConvOutOfBoundsBiasInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_TRANSPOSE_CONV,
                                            {0, 0, 0, 999}, {2});
      break;
    case VariableOpGuardCase::kCreateTransposeConvOutOfBoundsOutputShapeInput:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_TRANSPOSE_CONV,
                                            {131, 0, 0}, {2});
      break;
    case VariableOpGuardCase::kCreateSliceZeroOutputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_SLICE, {0, 0, 0},
                                            {});
      break;
    case VariableOpGuardCase::kCreateStridedSliceZeroOutputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_STRIDED_SLICE,
                                            {0, 1, 1, 1}, {});
      break;
    case VariableOpGuardCase::kCreateBatchMatMulZeroInputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_BATCH_MATMUL, {},
                                            {2});
      break;
    case VariableOpGuardCase::kCreateBatchMatMulZeroOutputs:
      run_synthetic_subgraph_create_op_case(BuiltinOperator_BATCH_MATMUL,
                                            {0, 0}, {});
      break;
    case VariableOpGuardCase::kCreateSdpaCustomZeroInputs:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_CUSTOM, {}, {2}, /*first_int_param=*/0,
          /*custom_name=*/"odml.scaled_dot_product_attention");
      break;
    case VariableOpGuardCase::kCreateSdpaCustomZeroOutputs:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_CUSTOM, {0, 0, 0}, {}, /*first_int_param=*/0,
          /*custom_name=*/"odml.scaled_dot_product_attention");
      break;
    case VariableOpGuardCase::kCreateOptionalNegativeInputInDefaultUnaryOp:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_RELU, {kTfLiteOptionalTensor}, {2});
      break;
    case VariableOpGuardCase::kCreateOptionalNegativeOutputInSubgraphCreate:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_RELU, {0}, {kTfLiteOptionalTensor});
      break;
    case VariableOpGuardCase::kCreateLeakyReluNullBuiltinData:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_LEAKY_RELU, {0}, {2}, /*first_int_param=*/0,
          /*custom_name=*/nullptr, /*null_builtin_data=*/true);
      break;
    case VariableOpGuardCase::kCreateGeluNullBuiltinData:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_GELU, {0}, {2}, /*first_int_param=*/0,
          /*custom_name=*/nullptr, /*null_builtin_data=*/true);
      break;
    case VariableOpGuardCase::kCreateSdpaCustomNegativeInitialDataSize:
      run_synthetic_subgraph_create_op_case(
          BuiltinOperator_CUSTOM, {0, 0, 0}, {2}, /*first_int_param=*/0,
          /*custom_name=*/"odml.scaled_dot_product_attention",
          /*null_builtin_data=*/false,
          /*custom_initial_data_size_override=*/-1);
      break;
  }
}

namespace {

void RunAliasedVarHandleConflictCase(const std::vector<int32_t>& assign_shape,
                                     TensorType assign_type,
                                     const std::vector<int32_t>& read_shape,
                                     TensorType read_type) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector({})),
  };
  const std::vector<int32_t> rshape = {1};
  auto make_tensor = [&](const std::vector<int32_t>& s, TensorType tt) {
    if (tt == TensorType_INT8 || tt == TensorType_UINT8) {
      return CreateTensor(
          builder, builder.CreateVector<int32_t>(s.data(), s.size()), tt,
          /*buffer=*/0, /*name=*/0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({1.0f}),
              builder.CreateVector<int64_t>({0})));
    }
    return CreateTensor(
        builder, builder.CreateVector<int32_t>(s.data(), s.size()), tt);
  };
  const std::vector<flatbuffers::Offset<Tensor>> tensors = {
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),
      make_tensor(assign_shape, assign_type),
      make_tensor(read_shape, read_type),
  };
  const auto vh0 = CreateOperator(
      builder, 0, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("v_alias"))
          .Union());
  const auto vh1 = CreateOperator(
      builder, 0, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("v_alias"))
          .Union());
  const auto av =
      CreateOperator(builder, 2, builder.CreateVector<int32_t>({0, 2}),
                     builder.CreateVector<int32_t>({}));
  const auto rv =
      CreateOperator(builder, 1, builder.CreateVector<int32_t>({1}),
                     builder.CreateVector<int32_t>({3}));
  const auto sg = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({2}), builder.CreateVector<int32_t>({3}),
      builder.CreateVector({vh0, vh1, av, rv}));
  builder.Finish(CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(opcodes.data(), opcodes.size()),
      builder.CreateVector(&sg, 1),
      builder.CreateString("AliasedConflictModel"),
      builder.CreateVector(buffers.data(), buffers.size())));

  auto d = NewXnnPackDelegateSupportingVariableOps();
  d->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  const Model* model = GetModel(builder.GetBufferPointer());
  std::unique_ptr<Interpreter> interp;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interp),
      kTfLiteOk);
  ASSERT_TRUE(interp);
  ASSERT_EQ(interp->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interp->ModifyGraphWithDelegate(d.get()), kTfLiteOk);
  EXPECT_EQ(interp->Invoke(), kTfLiteError);
}

}  // namespace

void VariableOpsTester::TestRejectsConflictingAliasedVarHandleShapes(
    TfLiteDelegate* delegate) const {
  (void)delegate;
  RunAliasedVarHandleConflictCase({1, 2, 2, 3}, TensorType_FLOAT32,
                                  {1, 5, 2, 3}, TensorType_FLOAT32);
}

void VariableOpsTester::TestRejectsConflictingAliasedVarHandleTypes(
    TfLiteDelegate* delegate) const {
  (void)delegate;
  RunAliasedVarHandleConflictCase({1, 2, 2, 12}, TensorType_UINT8,
                                  {1, 2, 2, 3}, TensorType_FLOAT32);
}

void VariableOpsTester::TestSingleSubgraphRejectsUninitializedRead(
    TfLiteDelegate* delegate) const {
  flatbuffers::FlatBufferBuilder ro_builder;
  const std::vector<flatbuffers::Offset<OperatorCode>> ro_opcodes = {
      CreateOperatorCode(ro_builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(ro_builder, BuiltinOperator_READ_VARIABLE),
  };
  const std::vector<flatbuffers::Offset<Buffer>> ro_buffers = {
      CreateBuffer(ro_builder, ro_builder.CreateVector({})),
  };
  const std::vector<int32_t> small_shape = {1, 2, 2, 3};
  const std::vector<int32_t> rshape = {1};
  const std::vector<flatbuffers::Offset<Tensor>> ro_tensors = {
      CreateTensor(
          ro_builder,
          ro_builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
          TensorType_RESOURCE),
      CreateTensor(ro_builder,
                   ro_builder.CreateVector<int32_t>(small_shape.data(),
                                                    small_shape.size()),
                   TensorType_FLOAT32),
  };
  const flatbuffers::Offset<Operator> ro_vh = CreateOperator(
      ro_builder, 0, ro_builder.CreateVector<int32_t>({}),
      ro_builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(ro_builder, ro_builder.CreateString("container"),
                             ro_builder.CreateString("never_assigned"))
          .Union());
  const flatbuffers::Offset<Operator> ro_rv = CreateOperator(
      ro_builder, 1, ro_builder.CreateVector<int32_t>({0}),
      ro_builder.CreateVector<int32_t>({1}));
  const flatbuffers::Offset<SubGraph> ro_sg = CreateSubGraph(
      ro_builder, ro_builder.CreateVector(ro_tensors.data(), ro_tensors.size()),
      ro_builder.CreateVector<int32_t>({}),
      ro_builder.CreateVector<int32_t>({1}),
      ro_builder.CreateVector({ro_vh, ro_rv}));
  ro_builder.Finish(CreateModel(
      ro_builder, TFLITE_SCHEMA_VERSION,
      ro_builder.CreateVector(ro_opcodes.data(), ro_opcodes.size()),
      ro_builder.CreateVector(&ro_sg, 1),
      ro_builder.CreateString("UninitializedReadVariableModel"),
      ro_builder.CreateVector(ro_buffers.data(), ro_buffers.size())));

  delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  const Model* ro_model = GetModel(ro_builder.GetBufferPointer());

  std::unique_ptr<Interpreter> cpu_interp;
  ASSERT_EQ(
      InterpreterBuilder(
          ro_model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &cpu_interp),
      kTfLiteOk);
  ASSERT_TRUE(cpu_interp);
  ASSERT_EQ(cpu_interp->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(cpu_interp->Invoke(), kTfLiteError);

  std::unique_ptr<Interpreter> ro_interp;
  ASSERT_EQ(
      InterpreterBuilder(
          ro_model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &ro_interp),
      kTfLiteOk);
  ASSERT_TRUE(ro_interp);
  ASSERT_EQ(ro_interp->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(ro_interp->ModifyGraphWithDelegate(delegate), kTfLiteOk);
  EXPECT_EQ(ro_interp->Invoke(), kTfLiteError);
}

void VariableOpsTester::TestMalformedVariableOpModelCase(
    TfLiteDelegate* delegate, MalformedVariableOpModelCase test_case) const {
  auto build_single_op_model = [](BuiltinOperator op_code,
                                  const std::vector<int32_t>& op_inputs,
                                  const std::vector<int32_t>& op_outputs,
                                  bool include_var_handle_opts) {
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
        CreateOperatorCode(builder, op_code),
    };
    const std::vector<flatbuffers::Offset<Buffer>> buffers{{
        CreateBuffer(builder, builder.CreateVector({})),
    }};
    const std::vector<int32_t> fshape = {1, 2, 2, 3};
    const std::vector<int32_t> rshape = {1};
    const std::vector<flatbuffers::Offset<Tensor>> tensors{{
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                     TensorType_FLOAT32),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                     TensorType_RESOURCE),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                     TensorType_FLOAT32),
    }};
    flatbuffers::Offset<Operator> op;
    if (include_var_handle_opts) {
      op = CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
          builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
          tflite::BuiltinOptions_VarHandleOptions,
          CreateVarHandleOptions(builder, builder.CreateString("c"),
                                 builder.CreateString("v"))
              .Union());
    } else {
      op = CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
          builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));
    }
    const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
        builder, builder.CreateVector(tensors.data(), tensors.size()),
        builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({2}),
        builder.CreateVector({op}));
    builder.Finish(CreateModel(
        builder, TFLITE_SCHEMA_VERSION,
        builder.CreateVector(operator_codes.data(), operator_codes.size()),
        builder.CreateVector(&subgraph, 1),
        builder.CreateString("MalformedSingleVariableOp"),
        builder.CreateVector(buffers.data(), buffers.size())));
    return std::vector<char>(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        reinterpret_cast<const char*>(builder.GetBufferPointer()) +
            builder.GetSize());
  };

  auto build_vh_plus_op_model = [](int32_t vh_out, BuiltinOperator second_op,
                                   const std::vector<int32_t>& second_inputs,
                                   const std::vector<int32_t>& second_outputs) {
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
        CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
        CreateOperatorCode(builder, second_op),
    };
    const std::vector<flatbuffers::Offset<Buffer>> buffers{{
        CreateBuffer(builder, builder.CreateVector({})),
    }};
    const std::vector<int32_t> fshape = {1, 2, 2, 3};
    const std::vector<int32_t> rshape = {1};
    const std::vector<flatbuffers::Offset<Tensor>> tensors{{
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                     TensorType_FLOAT32),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                     TensorType_RESOURCE),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                     TensorType_FLOAT32),
    }};
    const auto vh = CreateOperator(
        builder, /*opcode_index=*/0, builder.CreateVector<int32_t>({}),
        builder.CreateVector<int32_t>({vh_out}),
        tflite::BuiltinOptions_VarHandleOptions,
        CreateVarHandleOptions(builder, builder.CreateString("c"),
                               builder.CreateString("v"))
            .Union());
    const auto op2 = CreateOperator(
        builder, /*opcode_index=*/1,
        builder.CreateVector<int32_t>(second_inputs.data(),
                                      second_inputs.size()),
        builder.CreateVector<int32_t>(second_outputs.data(),
                                      second_outputs.size()));
    const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
        builder, builder.CreateVector(tensors.data(), tensors.size()),
        builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({2}),
        builder.CreateVector({vh, op2}));
    builder.Finish(CreateModel(
        builder, TFLITE_SCHEMA_VERSION,
        builder.CreateVector(operator_codes.data(), operator_codes.size()),
        builder.CreateVector(&subgraph, 1),
        builder.CreateString("MalformedPairedVariableOp"),
        builder.CreateVector(buffers.data(), buffers.size())));
    return std::vector<char>(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        reinterpret_cast<const char*>(builder.GetBufferPointer()) +
            builder.GetSize());
  };

  std::vector<char> buf;
  switch (test_case) {
    case MalformedVariableOpModelCase::kVarHandleZeroOutputs:
      buf = build_single_op_model(BuiltinOperator_VAR_HANDLE, {}, {}, true);
      break;
    case MalformedVariableOpModelCase::kReadVariableZeroInputs:
      buf =
          build_single_op_model(BuiltinOperator_READ_VARIABLE, {}, {2}, false);
      break;
    case MalformedVariableOpModelCase::kReadVariableZeroOutputs:
      buf =
          build_single_op_model(BuiltinOperator_READ_VARIABLE, {1}, {}, false);
      break;
    case MalformedVariableOpModelCase::kAssignVariableZeroInputs:
      buf =
          build_single_op_model(BuiltinOperator_ASSIGN_VARIABLE, {}, {}, false);
      break;
    case MalformedVariableOpModelCase::kAssignVariableOneInput:
      buf = build_single_op_model(BuiltinOperator_ASSIGN_VARIABLE, {1}, {},
                                  false);
      break;
    case MalformedVariableOpModelCase::kAssignVariableUnexpectedOutput:
      buf = build_vh_plus_op_model(1, BuiltinOperator_ASSIGN_VARIABLE, {1, 0},
                                   {2});
      break;
    case MalformedVariableOpModelCase::kAssignVariableNegativeValueInput:
      buf = build_vh_plus_op_model(1, BuiltinOperator_ASSIGN_VARIABLE, {1, -1},
                                   {});
      break;
    case MalformedVariableOpModelCase::kAssignVariableOutOfBoundsValueInput:
      buf = build_vh_plus_op_model(1, BuiltinOperator_ASSIGN_VARIABLE, {1, 0},
                                   {});
      break;
    case MalformedVariableOpModelCase::kReadVariableNegativeValueOutput:
      buf = build_vh_plus_op_model(1, BuiltinOperator_READ_VARIABLE, {1}, {-1});
      break;
    case MalformedVariableOpModelCase::kReadVariableOutOfBoundsValueOutput:
      buf = build_vh_plus_op_model(1, BuiltinOperator_READ_VARIABLE, {1}, {2});
      break;
    case MalformedVariableOpModelCase::
        kVarHandleAndReadVariableNegativeResourceWire:
      buf =
          build_vh_plus_op_model(-1, BuiltinOperator_READ_VARIABLE, {-1}, {2});
      break;
    case MalformedVariableOpModelCase::
        kVarHandleAndAssignVariableNegativeResourceWire:
      buf = build_vh_plus_op_model(-1, BuiltinOperator_ASSIGN_VARIABLE, {-1, 0},
                                   {});
      break;
  }

  const Model* model = GetModel(buf.data());
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);
  ASSERT_TRUE(interpreter);

  if (test_case ==
      MalformedVariableOpModelCase::kAssignVariableOutOfBoundsValueInput) {
    const auto* nr = interpreter->node_and_registration(1);
    ASSERT_NE(nr, nullptr);
    TfLiteNode* mutable_node = const_cast<TfLiteNode*>(&nr->first);
    ASSERT_EQ(mutable_node->inputs->size, 2);
    const int orig_idx = mutable_node->inputs->data[1];
    mutable_node->inputs->data[1] = 131;
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);
    EXPECT_EQ(interpreter->execution_plan().size(), 2u);
    mutable_node->inputs->data[1] = orig_idx;
    return;
  }
  if (test_case ==
      MalformedVariableOpModelCase::kReadVariableOutOfBoundsValueOutput) {
    const auto* nr = interpreter->node_and_registration(1);
    ASSERT_NE(nr, nullptr);
    TfLiteNode* mutable_node = const_cast<TfLiteNode*>(&nr->first);
    ASSERT_EQ(mutable_node->outputs->size, 1);
    const int orig_idx = mutable_node->outputs->data[0];
    mutable_node->outputs->data[0] = 131;
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);
    EXPECT_EQ(interpreter->execution_plan().size(), 2u);
    mutable_node->outputs->data[0] = orig_idx;
    return;
  }

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);
  const TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status == kTfLiteOk) {
    EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
  } else {
    EXPECT_EQ(alloc_status, kTfLiteError);
  }
}

void VariableOpsTester::TestTwoSubgraphsRejectsMismatchedShape(
    TfLiteDelegate* delegate) const {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };
  const std::vector<float> init_vals(12, 3.25f);
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(
          builder,
          builder.CreateVector(
              reinterpret_cast<const uint8_t*>(init_vals.data()),
              init_vals.size() * sizeof(float))),
  };
  const std::vector<int32_t> small_shape = {1, 2, 2, 3};
  const std::vector<int32_t> large_shape = {1, 5, 2, 3};
  const std::vector<int32_t> rshape = {1};

  const std::vector<flatbuffers::Offset<Tensor>> primary_tensors = {
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(large_shape.data(), large_shape.size()),
          TensorType_FLOAT32),
  };
  const flatbuffers::Offset<Operator> call_once_op = CreateOperator(
      builder, CALL_ONCE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({}), tflite::BuiltinOptions_CallOnceOptions,
      CreateCallOnceOptions(builder, /*init_subgraph_index=*/1).Union());
  const flatbuffers::Offset<Operator> primary_vh = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("v"))
          .Union());
  const flatbuffers::Offset<Operator> primary_rv = CreateOperator(
      builder, READ_VARIABLE, builder.CreateVector<int32_t>({0}),
      builder.CreateVector<int32_t>({1}));
  const flatbuffers::Offset<SubGraph> primary_sg = CreateSubGraph(
      builder,
      builder.CreateVector(primary_tensors.data(), primary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({1}),
      builder.CreateVector({call_once_op, primary_vh, primary_rv}));

  const std::vector<flatbuffers::Offset<Tensor>> secondary_tensors = {
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(small_shape.data(), small_shape.size()),
          TensorType_FLOAT32, /*buffer=*/1),
  };
  const flatbuffers::Offset<Operator> secondary_vh = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("v"))
          .Union());
  const flatbuffers::Offset<Operator> secondary_av = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({0, 1}),
      builder.CreateVector<int32_t>({}));
  const flatbuffers::Offset<SubGraph> secondary_sg = CreateSubGraph(
      builder,
      builder.CreateVector(secondary_tensors.data(), secondary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({}),
      builder.CreateVector({secondary_vh, secondary_av}));

  builder.Finish(CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector({primary_sg, secondary_sg}),
      builder.CreateString("MismatchedReadVariableModel"),
      builder.CreateVector(buffers.data(), buffers.size())));

  delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  const Model* model = GetModel(builder.GetBufferPointer());
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);
  ASSERT_TRUE(interpreter);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);
  EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
}

void VariableOpsTester::TestTwoSubgraphsRejectsUninitializedRead(
    TfLiteDelegate* delegate) const {
  // Build a 2-subgraph model where secondary CALL_ONCE subgraph initializes
  // ("container", "other_var") while primary delegated READ_VARIABLE reads
  // ("container", "never_assigned"). Reference CPU interpreter returns
  // kTfLiteError; delegated execution must also reject with kTfLiteError.
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<OperatorCode>> operator_codes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_CALL_ONCE),
  };
  const std::vector<float> init_vals(12, 1.5f);
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(
          builder,
          builder.CreateVector(
              reinterpret_cast<const uint8_t*>(init_vals.data()),
              init_vals.size() * sizeof(float))),
  };
  const std::vector<int32_t> shape = {1, 2, 2, 3};
  const std::vector<int32_t> rshape = {1};

  const std::vector<flatbuffers::Offset<Tensor>> primary_tensors = {
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
  };
  const auto call_once_op = CreateOperator(
      builder, CALL_ONCE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({}), tflite::BuiltinOptions_CallOnceOptions,
      CreateCallOnceOptions(builder, /*init_subgraph_index=*/1).Union());
  const auto primary_vh = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("never_assigned"))
          .Union());
  const auto primary_rv = CreateOperator(
      builder, READ_VARIABLE, builder.CreateVector<int32_t>({0}),
      builder.CreateVector<int32_t>({1}));
  const auto primary_sg = CreateSubGraph(
      builder,
      builder.CreateVector(primary_tensors.data(), primary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({1}),
      builder.CreateVector({call_once_op, primary_vh, primary_rv}));

  const std::vector<flatbuffers::Offset<Tensor>> secondary_tensors = {
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32, /*buffer=*/1),
  };
  const auto secondary_vh = CreateOperator(
      builder, VAR_HANDLE, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("container"),
                             builder.CreateString("other_var"))
          .Union());
  const auto secondary_av = CreateOperator(
      builder, ASSIGN_VARIABLE, builder.CreateVector<int32_t>({0, 1}),
      builder.CreateVector<int32_t>({}));
  const auto secondary_sg = CreateSubGraph(
      builder,
      builder.CreateVector(secondary_tensors.data(), secondary_tensors.size()),
      builder.CreateVector<int32_t>({}), builder.CreateVector<int32_t>({}),
      builder.CreateVector({secondary_vh, secondary_av}));

  builder.Finish(CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector({primary_sg, secondary_sg}),
      builder.CreateString("TwoSubgraphsUninitializedReadModel"),
      builder.CreateVector(buffers.data(), buffers.size())));

  const Model* model = GetModel(builder.GetBufferPointer());
  std::unique_ptr<Interpreter> cpu_interp;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &cpu_interp),
      kTfLiteOk);
  ASSERT_TRUE(cpu_interp);
  ASSERT_EQ(cpu_interp->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(cpu_interp->Invoke(), kTfLiteError);

  delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  std::unique_ptr<Interpreter> delegated_interp;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegated_interp),
      kTfLiteOk);
  ASSERT_TRUE(delegated_interp);
  ASSERT_EQ(delegated_interp->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(delegated_interp->ModifyGraphWithDelegate(delegate), kTfLiteOk);
  EXPECT_EQ(delegated_interp->Invoke(), kTfLiteError);
}

void VariableOpsTester::
    TestConflictingLocalResourceVarHandleFallsBackWithoutCorruptingFirstVar(
        TfLiteDelegate* delegate) const {
  // Single subgraph where local RESOURCE tensor 0 is first bound to ("c","v1")
  // and assigned input A, then rebound by a second VAR_HANDLE to ("c","v2")
  // and assigned input B, after which ("c","v1") is read via local RESOURCE
  // tensor 1 into output. Reference CPU semantics: output == A (not B).
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCode(builder, BuiltinOperator_VAR_HANDLE),
      CreateOperatorCode(builder, BuiltinOperator_READ_VARIABLE),
      CreateOperatorCode(builder, BuiltinOperator_ASSIGN_VARIABLE),
  };
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector({})),
  };
  const std::vector<int32_t> fshape = {1, 2, 2, 3};
  const std::vector<int32_t> rshape = {1};
  const std::vector<flatbuffers::Offset<Tensor>> tensors = {
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),  // 0: conflicted local resource
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(rshape.data(), rshape.size()),
                   TensorType_RESOURCE),  // 1: clean handle for v1
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                   TensorType_FLOAT32),   // 2: input A
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                   TensorType_FLOAT32),   // 3: input B
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(fshape.data(), fshape.size()),
                   TensorType_FLOAT32),   // 4: output (must equal A)
  };
  const auto vh_v1_on_r0 = CreateOperator(
      builder, 0, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("c"),
                             builder.CreateString("v1"))
          .Union());
  const auto av_a =
      CreateOperator(builder, 2, builder.CreateVector<int32_t>({0, 2}),
                     builder.CreateVector<int32_t>({}));
  const auto vh_v2_on_r0 = CreateOperator(
      builder, 0, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({0}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("c"),
                             builder.CreateString("v2"))
          .Union());
  const auto av_b =
      CreateOperator(builder, 2, builder.CreateVector<int32_t>({0, 3}),
                     builder.CreateVector<int32_t>({}));
  const auto vh_v1_on_r1 = CreateOperator(
      builder, 0, builder.CreateVector<int32_t>({}),
      builder.CreateVector<int32_t>({1}),
      tflite::BuiltinOptions_VarHandleOptions,
      CreateVarHandleOptions(builder, builder.CreateString("c"),
                             builder.CreateString("v1"))
          .Union());
  const auto rv_v1 =
      CreateOperator(builder, 1, builder.CreateVector<int32_t>({1}),
                     builder.CreateVector<int32_t>({4}));
  const auto sg = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({2, 3}), builder.CreateVector<int32_t>({4}),
      builder.CreateVector(
          {vh_v1_on_r0, av_a, vh_v2_on_r0, av_b, vh_v1_on_r1, rv_v1}));
  builder.Finish(CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(opcodes.data(), opcodes.size()),
      builder.CreateVector(&sg, 1),
      builder.CreateString("ReboundLocalResourceHandleModel"),
      builder.CreateVector(buffers.data(), buffers.size())));

  const Model* model = GetModel(builder.GetBufferPointer());
  std::unique_ptr<Interpreter> cpu_interp;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &cpu_interp),
      kTfLiteOk);
  ASSERT_TRUE(cpu_interp);
  ASSERT_EQ(cpu_interp->AllocateTensors(), kTfLiteOk);

  delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  std::unique_ptr<Interpreter> del_interp;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &del_interp),
      kTfLiteOk);
  ASSERT_TRUE(del_interp);
  ASSERT_EQ(del_interp->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(del_interp->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  for (int i = 0; i < 12; ++i) {
    cpu_interp->typed_input_tensor<float>(0)[i] = 10.0f + i;
    cpu_interp->typed_input_tensor<float>(1)[i] = -50.0f - i;
    del_interp->typed_input_tensor<float>(0)[i] = 10.0f + i;
    del_interp->typed_input_tensor<float>(1)[i] = -50.0f - i;
  }
  ASSERT_EQ(cpu_interp->Invoke(), kTfLiteOk);
  ASSERT_EQ(del_interp->Invoke(), kTfLiteOk);
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(del_interp->typed_output_tensor<float>(0)[i],
              cpu_interp->typed_output_tensor<float>(0)[i])
        << "idx=" << i;
  }
}

}  // namespace xnnpack
}  // namespace tflite
