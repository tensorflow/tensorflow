/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/subgraph_test_util.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

// Forward declaration for op kernels.
namespace ops {
namespace custom {
namespace random_int {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 0);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
  outputSize->data[0] = 1;
  return context->ResizeTensor(context, output, outputSize);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  std::random_device rd;
  std::uniform_int_distribution<int> dist(1, 32768);
  output.data.i32[0] = dist(rd);
  return kTfLiteOk;
}

}  // namespace random_int

TfLiteRegistration* Register_RANDOM_INT() {
  static TfLiteRegistration r = {nullptr, nullptr, random_int::Prepare,
                                 random_int::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops

namespace subgraph_test_util {

namespace {

void AddTileNode(Subgraph* subgraph, int input0, int input1, int output) {
  int node_index;
  auto* tile_reg = ops::builtin::Register_TILE();
  tile_reg->builtin_code = kTfLiteBuiltinTile;
  subgraph->AddNodeWithParameters({input0, input1}, {output}, {}, nullptr, 0,
                                  nullptr, tile_reg, &node_index);
}

void AddFlexNode(Subgraph* subgraph, int input_tensor, int output_tensor) {
  auto prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor& input = context->tensors[node->inputs->data[0]];
    TfLiteTensor& output = context->tensors[node->outputs->data[0]];
    TfLiteArrayUniquePtr<int> shape =
        BuildTfLiteArray(input.dims->size, input.dims->data);
    return context->ResizeTensor(context, &output, shape.release());
  };
  auto eval = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor& input = context->tensors[node->inputs->data[0]];
    TfLiteTensor& output = context->tensors[node->outputs->data[0]];
    memcpy(output.data.data, input.data.data, input.bytes);
    return kTfLiteOk;
  };

  TfLiteRegistration reg = {nullptr, nullptr, prepare, eval};
  reg.builtin_code = BuiltinOperator_CUSTOM;
  reg.custom_name = "Flex";

  int node_index;
  ASSERT_EQ(
      subgraph->AddNodeWithParameters({input_tensor}, {output_tensor}, {},
                                      nullptr, 0, nullptr, &reg, &node_index),
      kTfLiteOk);
}

void AddReshapeNode(Subgraph* subgraph, int input0, int input1, int output) {
  int node_index;
  TfLiteReshapeParams* reshape_params = reinterpret_cast<TfLiteReshapeParams*>(
      calloc(1, sizeof(TfLiteReshapeParams)));
  auto* reshape_reg = ops::builtin::Register_RESHAPE();
  reshape_reg->builtin_code = kTfLiteBuiltinReshape;
  ASSERT_EQ(subgraph->AddNodeWithParameters({input0, input1}, {output}, {},
                                            nullptr, 0, reshape_params,
                                            reshape_reg, &node_index),
            kTfLiteOk);
}

void AddOffsetAddNode(Subgraph* subgraph, int input0, int input1, int output) {
  auto prepare = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor& input0 = context->tensors[node->inputs->data[0]];
    TfLiteTensor& output = context->tensors[node->outputs->data[0]];
    TfLiteIntArray* shape = TfLiteIntArrayCopy(input0.dims);
    return context->ResizeTensor(context, &output, shape);
  };

  auto invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor& input0 = context->tensors[node->inputs->data[0]];
    const TfLiteTensor& input1 = context->tensors[node->inputs->data[1]];
    TfLiteTensor& output = context->tensors[node->outputs->data[0]];
    int num_elements = input0.dims->data[0];
    const int kOffset = 1;
    const int* i0 = static_cast<int*>(input0.data.data);
    const int* i1 = static_cast<int*>(input1.data.data);
    int* o = static_cast<int*>(output.data.data);
    for (int i = 0; i < num_elements; ++i) {
      int input0_pos = (i + kOffset) % num_elements;
      o[i] = i0[input0_pos] + i1[i];
    }
    return kTfLiteOk;
  };

  int node_index;
  TfLiteRegistration offset_add_reg = {/*Init=*/nullptr, /*Free=*/nullptr,
                                       prepare, invoke};
  offset_add_reg.builtin_code = BuiltinOperator_CUSTOM;
  offset_add_reg.custom_name = "OffsetAdd";
  offset_add_reg.inplace_operator = kTfLiteInplaceOpInput1Shared;
  subgraph->AddNodeWithParameters({input0, input1}, {output}, {}, nullptr, 0,
                                  nullptr, &offset_add_reg, &node_index);
}

void AddAddNode(Subgraph* subgraph, int input0, int input1, int output) {
  int node_index;
  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(calloc(1, sizeof(TfLiteAddParams)));
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({input0, input1}, {output}, {}, nullptr, 0,
                                  add_params, add_reg, &node_index);
}

// Add a DYNAMIC_UPDATE_SLICE node to the subgraph.
void AddDynamicUpdateSliceNode(Subgraph* subgraph, int input0, int input1,
                               int input2, int output) {
  int node_index;
  auto* reg = ops::builtin::Register_DYNAMIC_UPDATE_SLICE();
  reg->builtin_code = kTfLiteBuiltinDynamicUpdateSlice;
  subgraph->AddNodeWithParameters({input0, input1, input2}, {output}, {},
                                  nullptr, 0, nullptr, reg, &node_index);
}
}  // namespace

void Setup1DTensor(Subgraph* subgraph, int tensor_index, TfLiteType type) {
  int dim = 1;
  ASSERT_EQ(subgraph->SetTensorParametersReadWrite(tensor_index, type, "", 1,
                                                   &dim, {}, false),
            kTfLiteOk);
}

void SetupTensor(Subgraph* subgraph, int tensor_index, TfLiteType type) {
  ASSERT_EQ(subgraph->SetTensorParametersReadWrite(tensor_index, type, "", 0,
                                                   nullptr, {}, false),
            kTfLiteOk);
}

SubgraphBuilder::~SubgraphBuilder() {
  for (auto buffer : buffers_) {
    free(buffer);
  }
}

void SubgraphBuilder::BuildInplaceDynamicUpdateSliceSubgraph(
    Subgraph& subgraph, bool multiple_consumers) {
  enum {
    kInput0,
    kInput1,
    kInput2,
    kConstRhs,
    kOutput,
    kIntermediate0,
    kIntermediate1,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput0, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput}), kTfLiteOk);
  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(&subgraph, i, kTfLiteInt32);
  }

  //   kInput0 --> +---+
  //               |ADD| --> kIntermediate0 --> +---+
  // kConstRhs --> +---+       |    kInput1 --> |DUS| --> kIntermediate1
  //     |                     |    kInput2 --> +---+           |
  //     |                     |                                +----> +---+
  //     |if one consumer      |if multiple consumers                  |ADD|
  //     +---------------------+-------------------------------------> +---+
  CreateConstantTensor(&subgraph, kConstRhs, {1}, {1});
  AddAddNode(&subgraph, kInput0, kConstRhs, kIntermediate0);
  AddDynamicUpdateSliceNode(&subgraph, kIntermediate0, kInput1, kInput2,
                            kIntermediate1);
  AddAddNode(&subgraph, kIntermediate1,
             multiple_consumers ? kIntermediate0 : kConstRhs, kOutput);
}

void SubgraphBuilder::BuildInputDynamicUpdateSliceSubgraph(Subgraph& subgraph) {
  enum {
    kInput0,
    kInput1,
    kInput2,
    kConstRhs,
    kOutput,
    kIntermediate0,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput0, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput}), kTfLiteOk);
  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(&subgraph, i, kTfLiteInt32);
  }

  // kInput0 --> +---+
  // kInput1 --> |DUS| --> kIntermediate0 --> +---+
  // kInput2 --> +---+                        |ADD| --> kOutput
  //                            kConstRhs --> +---+
  CreateConstantTensor(&subgraph, kConstRhs, {1}, {1});
  AddDynamicUpdateSliceNode(&subgraph, kInput0, kInput1, kInput2,
                            kIntermediate0);
  AddAddNode(&subgraph, kIntermediate0, kConstRhs, kOutput);
}

void SubgraphBuilder::BuildOutputNotConsumedSubgraph(Subgraph& subgraph) {
  enum {
    kInput0,
    kInput1,
    kInput2,
    kOutput0,
    kOutput1,
    kConstRhs,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph.AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph.SetInputs({kInput0, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({kOutput0, kOutput1, kConstRhs}), kTfLiteOk);
  for (int i = 0; i < kTensorCount; ++i) {
    Setup1DTensor(&subgraph, i, kTfLiteInt32);
  }

  // kInput0 --> +---+
  // kInput1 --> |DUS| --> kIntermediate0 --> +---+
  // kInput2 --> +---+                        |ADD| --> kOutput
  //                            kConstRhs --> +---+
  CreateConstantTensor(&subgraph, kConstRhs, {1}, {1});
  AddAddNode(&subgraph, kInput0, kInput1, kOutput0);
  AddTileNode(&subgraph, kInput0, kInput2, kOutput1);
}

void SubgraphBuilder::BuildXNNPACKSubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue,
    kOutputCounter,
    kOutputValue,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteFloat32);
  }

  AddAddNode(subgraph, kInputCounter, kInputValue, kIntermediateTensor0);
  AddAddNode(subgraph, kInputCounter, kInputValue, kIntermediateTensor1);
  AddAddNode(subgraph, kIntermediateTensor0, kIntermediateTensor1,
             kOutputCounter);
  AddAddNode(subgraph, kIntermediateTensor0, kIntermediateTensor1,
             kOutputValue);
}

void SubgraphBuilder::BuildInputIsOutputSubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue0,
    kInputOutput,
    kOutputCounter,
    kOutputValue0,
    kConstRhs,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue0, kInputOutput}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue0, kInputOutput}),
            kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstRhs, {1}, {1});

  AddAddNode(subgraph, kInputCounter, kConstRhs, kOutputCounter);
  AddAddNode(subgraph, kInputValue0, kInputOutput, kOutputValue0);
}

void SubgraphBuilder::BuildInputIsDifferentOutputSubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue,
    kOutputCounter,
    kOutputValue,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kInputValue, kOutputValue}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }

  AddAddNode(subgraph, kInputCounter, kInputValue, kOutputValue);
}

void SubgraphBuilder::BuildFlexOutputSubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue,
    kOutputCounter,
    kOutputValue,
    kConstRhs,
    kIntermediateTensor,
    kTensorCount
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstRhs, {1}, {1});

  AddAddNode(subgraph, kInputCounter, kConstRhs, kOutputCounter);
  AddAddNode(subgraph, kConstRhs, kInputValue, kIntermediateTensor);
  AddFlexNode(subgraph, kIntermediateTensor, kOutputValue);
}

void SubgraphBuilder::BuildCounterOnlySubgraph(Subgraph* subgraph) {
  enum { kInputCounter, kOutputCounter, kConstRhs, kTensorCount };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstRhs, {1}, {1});

  AddAddNode(subgraph, kInputCounter, kConstRhs, kOutputCounter);
}

void SubgraphBuilder::BuildAddSubgraph(Subgraph* subgraph,
                                       const TfLiteType operand_type) {
  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_ADD, kTfLiteBuiltinAdd,
                        params, operand_type, operand_type, operand_type);
}

void SubgraphBuilder::BuildStablehloAddSubgraph(Subgraph* subgraph,
                                                const TfLiteType operand_type) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_STABLEHLO_ADD,
                        kTfLiteBuiltinStablehloAdd, nullptr, operand_type,
                        operand_type, operand_type);
}

// This body subgraph has arena and dynamic output tensors which are not in
// place to verify that body subgraph outputs are written directly to node
// outputs. It also has inplace dynamic and arena outputs.
void SubgraphBuilder::BuildAllInplaceScenariosSubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue0,
    kInputValue1,
    kInputValue2,
    kOutputCounter,
    kOutputValue0,
    kOutputValue1,
    kOutputValue2,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kInputOutputTensor,
    kTensorCount
  };
  // kInputCounter --> +---+
  //                   |Add| --> kOutputCounter
  // kConstRhs ------> +---+
  //
  // kInputValue1 --> +------+
  //                  | TILE | -> kOutputValue1 ------|
  // kInputCounter -> +------+                        v
  //                                                  +-----+
  // kInputValue0 --> +-----+                         | Add |->kOutputValue0
  //                  | Add | -> kIntermediateTensor->+-----+
  // kConstRhs -----> +-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue0, kInputValue1,
                                 kInputValue2, kInputOutputTensor}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue0, kOutputValue1,
                                  kOutputValue2, kInputOutputTensor}),
            kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }

  CreateConstantTensor(subgraph, kInputOutputTensor, {1}, {1});

  // Arena, not in place.
  AddAddNode(subgraph, kInputCounter, kInputOutputTensor, kOutputCounter);

  // Arena, in place.
  AddAddNode(subgraph, kInputValue0, kInputOutputTensor, kIntermediateTensor0);
  AddAddNode(subgraph, kIntermediateTensor0, kInputOutputTensor, kOutputValue0);

  // Dynamic, not in place.
  AddTileNode(subgraph, kInputValue1, kInputCounter, kOutputValue1);

  // Dynamic, in place.
  AddTileNode(subgraph, kInputValue2, kInputCounter, kIntermediateTensor1);
  AddAddNode(subgraph, kIntermediateTensor1, kInputOutputTensor, kOutputValue2);
}

void SubgraphBuilder::BuildDynamicOpTriggersAllocationOfUnsedInputSubgraph(
    Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue0,
    kInputValue1,
    kOutputCounter,
    kOutputValue0,
    kOutputValue1,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kConstRhs,
    kTensorCount
  };
  // kInputCounter --> +---+
  //                   |Add| --> kOutputCounter
  // kConstRhs ------> +---+
  //
  // kInputValue1 --> +------+
  //                  | TILE | -> kOutputValue1 ------|
  // kInputCounter -> +------+                        v
  //                                                  +-----+
  // kInputValue0 --> +-----+                         | Add |->kOutputValue0
  //                  | Add | -> kIntermediateTensor->+-----+
  // kConstRhs -----> +-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue0, kInputValue1}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kOutputCounter, kOutputValue0, kOutputValue1}),
      kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }

  CreateConstantTensor(subgraph, kConstRhs, {1}, {1});

  AddAddNode(subgraph, kInputCounter, kConstRhs, kOutputCounter);
  AddTileNode(subgraph, kInputValue1, kInputCounter, kOutputValue1);
  AddAddNode(subgraph, kInputValue0, kConstRhs, kIntermediateTensor0);
  AddAddNode(subgraph, kIntermediateTensor0, kOutputValue1, kOutputValue0);
}

void SubgraphBuilder::BuildBinaryOpSubgraph(
    Subgraph* subgraph, TfLiteRegistration* (*Register_OP)(),
    const TfLiteBuiltinOperator builtin_code, void* const params,
    const TfLiteType input1_type, const TfLiteType input2_type,
    const TfLiteType output_type) {
  enum { kInput1, kInput2, kOutput, kTensorCount };
  // kInput1(0) --> +---+
  //                | OP| --> kOutput(2)
  // kInput2(1) --> +---+

  ASSERT_NE(Register_OP, nullptr);

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, input1_type);
  SetupTensor(subgraph, kInput2, input2_type);
  SetupTensor(subgraph, kOutput, output_type);

  TfLiteRegistration* reg = Register_OP();
  reg->builtin_code = builtin_code;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, reg, &node_index);
}

void SubgraphBuilder::BuildMaximumSubgraph(Subgraph* subgraph,
                                           const TfLiteType operand_type) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_MAXIMUM,
                        kTfLiteBuiltinMaximum, /*params=*/nullptr,
                        /*input1_type=*/operand_type,
                        /*input2_type=*/operand_type,
                        /*output_type=*/operand_type);
}

void SubgraphBuilder::BuildStablehloMaximumSubgraph(
    Subgraph* subgraph, const TfLiteType operand_type) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_STABLEHLO_MAXIMUM,
                        kTfLiteBuiltinStablehloMaximum, nullptr, operand_type,
                        operand_type, operand_type);
}

void SubgraphBuilder::BuildMinimumSubgraph(Subgraph* subgraph,
                                           const TfLiteType operand_type) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_MINIMUM,
                        kTfLiteBuiltinMinimum, /*params=*/nullptr,
                        /*input1_type=*/operand_type,
                        /*input2_type=*/operand_type,
                        /*output_type=*/operand_type);
}

void SubgraphBuilder::BuildStablehloMinimumSubgraph(
    Subgraph* subgraph, const TfLiteType operand_type) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_STABLEHLO_MINIMUM,
                        kTfLiteBuiltinStablehloMinimum, nullptr, operand_type,
                        operand_type, operand_type);
}

void SubgraphBuilder::BuildLogicalOrSubgraph(Subgraph* subgraph) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_LOGICAL_OR,
                        kTfLiteBuiltinLogicalOr, /*params=*/nullptr,
                        /*input1_type=*/kTfLiteBool,
                        /*input2_type=*/kTfLiteBool,
                        /*output_type=*/kTfLiteBool);
}

void SubgraphBuilder::BuildLogicalAndSubgraph(Subgraph* subgraph) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_LOGICAL_AND,
                        kTfLiteBuiltinLogicalAnd, /*params=*/nullptr,
                        /*input1_type=*/kTfLiteBool,
                        /*input2_type=*/kTfLiteBool,
                        /*output_type=*/kTfLiteBool);
}

void SubgraphBuilder::BuildOutputIsSecondInputSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kTensorCount = 2;
  // kInput1(0) --> x
  //                    | --> kOutput(2)
  // kInput2(1) --> ----^

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kInput2}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
}

// Build a subgraph with an mul op. Helper function for testing.
void SubgraphBuilder::BuildMulSubgraph(Subgraph* subgraph,
                                       TfLiteType operand_type) {
  TfLiteMulParams* params =
      reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  params->activation = kTfLiteActNone;
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_MUL, kTfLiteBuiltinMul,
                        params, /*input1_type=*/operand_type,
                        /*input2_type=*/operand_type,
                        /*output_type=*/operand_type);
}

void SubgraphBuilder::BuildStablehloMulSubgraph(Subgraph* subgraph,
                                                const TfLiteType operand_type) {
  BuildBinaryOpSubgraph(subgraph, ops::builtin::Register_STABLEHLO_MULTIPLY,
                        kTfLiteBuiltinStablehloMultiply, nullptr, operand_type,
                        operand_type, operand_type);
}

// Build a subgraph with a pad op. Helper function for testing.
void SubgraphBuilder::BuildPadSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput = 2;
  const int kTensorCount = 3;
  // kInput1(0) --> +---+
  //                |PAD| --> kOutput(2)
  // kInput2(1) --> +---+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLitePadParams* params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLitePadParams)));
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kInput2}, {kOutput}, {}, nullptr, 0,
                                  params, pad_reg, &node_index);
}

void SubgraphBuilder::BuildIfSubgraph(Subgraph* subgraph) {
  const int kCondInput = 0;
  const int kInput1 = 1;
  const int kInput2 = 2;
  const int kOutput = 3;
  const int kTensorCount = 4;

  // kCondInput(0) --> +----+
  // kInput1(1)  ----> | IF | --> kOutput(3)
  // kInput2(2)  ----> +----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kCondInput, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kCondInput, kTfLiteBool);
  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteIfParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters({kCondInput, kInput1, kInput2}, {kOutput}, {},
                                  nullptr, 0, params, if_reg, &node_index);
}

void SubgraphBuilder::BuildLargeLessEqualCondSubgraph(Subgraph* subgraph,
                                                      int rhs, int num_inputs) {
  const int kOutput = 0;
  const int kConstRhs = 1;
  const int kInput0 = 2;
  // kInput1(0) ----> +------------+
  //                  | LESS_EQUAL | --> kOutput(2)
  // kConstRhs(3) --> +------------+
  //
  // kInput2(1) --> (unused)

  int tensor_count = 3 + num_inputs;
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(tensor_count, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = kInput0 + i;
  }
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  for (int i = 0; i < num_inputs; ++i) {
    SetupTensor(subgraph, kInput0 + i, kTfLiteInt32);
  }
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantTensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kInput0, kConstRhs}, {kOutput}, {}, nullptr,
                                  0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildOffsetAddSharing(Subgraph* subgraph) {
  enum {
    kInput0,
    kInput1,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kIntermediateTensor2,
    kOutput,
    kConstRhs,
    kTensorCount,
  };
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput0, kInput1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor<int>(subgraph, kConstRhs, {1}, {1});
  // Consume input so that following ops can share tensor memory.
  AddAddNode(subgraph, kInput0, kInput1, kIntermediateTensor0);
  AddAddNode(subgraph, kInput0, kInput1, kIntermediateTensor1);
  // Input1 may be shared but not input0.
  AddOffsetAddNode(subgraph, kIntermediateTensor0, kIntermediateTensor1,
                   kIntermediateTensor2);
  AddAddNode(subgraph, kIntermediateTensor2, kConstRhs, kOutput);
}

void SubgraphBuilder::BuildBroadcastingSubgraph(Subgraph* subgraph) {
  enum {
    kInput0,
    kInput1,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kIntermediateTensor2,
    kIntermediateTensor3,
    kOutput,
    kConstRhs,
    kTensorCount,
  };
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput0, kInput1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor<int>(subgraph, kConstRhs, {1}, {1});
  // Consume input so that following ops can share tensor memory.
  AddAddNode(subgraph, kInput0, kInput1, kIntermediateTensor0);
  // Sharing is not possible as there is more than one consumer of
  // kIntermediateTensor0.
  AddAddNode(subgraph, kIntermediateTensor0, kIntermediateTensor0,
             kIntermediateTensor1);
  // Broadcasting ADD with sharing input1. kIntermediateTensor2 will share the
  // same memory as kIntermediateTensor0 and kIntermediateTensor1.
  AddAddNode(subgraph, kConstRhs, kIntermediateTensor1, kIntermediateTensor2);
  AddAddNode(subgraph, kIntermediateTensor2, kConstRhs, kIntermediateTensor3);
  // Consume this output to allow sharing.
  AddAddNode(subgraph, kIntermediateTensor3, kConstRhs, kOutput);
}

void SubgraphBuilder::BuildInplaceOpSubgraph(Subgraph* subgraph) {
  enum {
    kInput0,
    kInput1,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kOutput,
    kConstRhs,
    kTensorCount,
  };
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput0, kInput1}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor<int>(subgraph, kConstRhs, {1}, {2});
  // Consume input as Reshape cannot share subgraph input.
  AddAddNode(subgraph, kInput0, kInput1, kIntermediateTensor0);
  // Shape tensor is constant so that output is arena allocated.
  AddReshapeNode(subgraph, kIntermediateTensor0, kConstRhs,
                 kIntermediateTensor1);
  // Consume output of Reshape as subgraph outputs cannot be shared.
  AddAddNode(subgraph, kIntermediateTensor1, kInput1, kOutput);
}

void SubgraphBuilder::BuildFloatLessCondSubgraph(Subgraph* subgraph,
                                                 float rhs) {
  enum {
    kInput1,
    kInput2,
    kOutput,
    kConstRhs,
    kTensorCount,
  };

  // kInput1(0) ----> +------------+
  //                  | LESS_EQUAL | --> kOutput(2)
  // kConstRhs(3) --> +------------+
  //
  // kInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }

  auto* le_reg = ops::builtin::Register_LESS();
  le_reg->builtin_code = kTfLiteBuiltinLess;

  CreateConstantTensor<float>(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kConstRhs}, {kOutput}, {}, nullptr,
                                  0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildLessEqualCondSubgraph(Subgraph* subgraph, int rhs) {
  enum {
    kInput1,
    kInput2,
    kOutput,
    kConstRhs,
    kTensorCount,
  };

  // kInput1(0) ----> +------------+
  //                  | LESS_EQUAL | --> kOutput(2)
  // kConstRhs(3) --> +------------+
  //
  // kInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  for (int i = 0; i < kTensorCount - 1; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantTensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kInput1, kConstRhs}, {kOutput}, {}, nullptr,
                                  0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildLargeBodySubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue,
    kOutputCounter,
    kOutputValue,
    kConstStep,
    kConstSum,
    kIntermediateTensor0,
    kTensorCount,
  };

  // kInputCounter(0) --> +-----+
  //                      | ADD | -> kIntermediateTensor0(6)
  // kInputValue(1) ----> +-----+               |
  //                                            v
  //                                          +-----+
  //                                          | SUM | --> kOutputCounter(2)
  // kConstSum(4)   ----------------------->  +-----+
  //                                             |
  //                                             v
  //                                           +-----+
  //                                           | ADD | --> kOutputValue(3)
  // kConstStep(4)  ---------------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstSum, {1}, {-1});
  CreateConstantTensor(subgraph, kConstStep, {1}, {4});

  int node_index;
  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  auto* add_reg0 = ops::builtin::Register_ADD();
  add_reg0->builtin_code = kTfLiteBuiltinAdd;
  auto* add_reg1 = ops::builtin::Register_ADD();
  add_reg1->builtin_code = kTfLiteBuiltinAdd;

  subgraph->AddNodeWithParameters({kInputCounter, kInputValue},
                                  {kIntermediateTensor0}, {}, nullptr, 0,
                                  params, add_reg0, &node_index);
  auto* sum_reg = ops::builtin::Register_SUM();
  sum_reg->builtin_code = kTfLiteBuiltinSum;
  TfLiteReducerParams* sum_params = reinterpret_cast<TfLiteReducerParams*>(
      calloc(1, sizeof(TfLiteReducerParams)));
  subgraph->AddNodeWithParameters({kInputValue, kConstSum}, {kOutputCounter},
                                  {}, nullptr, 0, sum_params, sum_reg,
                                  &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  subgraph->AddNodeWithParameters({kIntermediateTensor0, kConstStep},
                                  {kOutputValue}, {}, nullptr, 0, params,
                                  add_reg1, &node_index);
}

void SubgraphBuilder::BuildDynamicBodySubgraphWithAliases(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue0,
    kInputValue1,
    kInputValue2,
    kInputValue3,
    kOutputCounter,
    kOutputValue0,
    kOutputValue1,
    kOutputValue2,
    kOutputValue3,
    kConstSum0,
    kConstSum1,
    kConstSum2,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kIntermediateTensor2,
    kTensorCount,
  };

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue0, kInputValue1,
                                 kInputValue2, kInputValue3}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue0, kOutputValue1,
                                  kOutputValue2, kOutputValue3}),
            kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }

  CreateConstantTensor(subgraph, kConstSum0, {1}, {1});
  CreateConstantTensor(subgraph, kConstSum1, {1}, {2});
  CreateConstantTensor(subgraph, kConstSum2, {1}, {3});

  AddAddNode(subgraph, kInputCounter, kConstSum0, kOutputCounter);
  AddAddNode(subgraph, kInputValue0, kInputValue1, kIntermediateTensor0);
  AddAddNode(subgraph, kInputValue2, kInputValue3, kIntermediateTensor1);
  AddAddNode(subgraph, kIntermediateTensor0, kIntermediateTensor1,
             kIntermediateTensor2);
  AddAddNode(subgraph, kIntermediateTensor2, kConstSum0, kOutputValue0);
  AddAddNode(subgraph, kIntermediateTensor2, kConstSum1, kOutputValue1);
  AddAddNode(subgraph, kIntermediateTensor2, kConstSum2, kOutputValue2);
  AddAddNode(subgraph, kIntermediateTensor2, kConstSum2, kOutputValue3);
}

void SubgraphBuilder::BuildDynamicIncreasingSizeSubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue0,
    kInputValue1,
    kInputValue2,
    kOutputCounter,
    kOutputValue0,
    kOutputValue1,
    kOutputValue2,
    kConstSum,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kIntermediateTensor2,
    kTensorCount,
  };

  // kInputCounter -----> +-----+
  //                      | ADD | -> kIntermediateTensor0 -> Add ->
  //                      kOutputCounter
  // kConstsum ---------> +-----+
  //
  // kInputValue0 ------> +-----+
  //                      | ADD | -> kIntermediateTensor0(6)
  // kInputValue1   ----> +-----+               |
  //                                            v
  //                                          +-----+
  //                                          | ADD | --> kOutputValue0
  // kConstSum(4)   ----------------------->  +-----+
  //                                             |
  //                                             v
  //                                           +-----+
  //                                           | PAD | --> kOutputValue1
  // kConstStep(4)  ---------------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs(
                {kInputCounter, kInputValue0, kInputValue1, kInputValue2}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(
                {kOutputCounter, kOutputValue0, kOutputValue1, kOutputValue2}),
            kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstSum, {1}, {1});

  AddAddNode(subgraph, kInputCounter, kConstSum, kOutputCounter);

  AddTileNode(subgraph, kInputValue0, kInputCounter, kIntermediateTensor0);
  AddAddNode(subgraph, kInputValue1, kConstSum, kIntermediateTensor1);
  AddAddNode(subgraph, kInputValue2, kConstSum, kIntermediateTensor2);

  AddAddNode(subgraph, kIntermediateTensor0, kConstSum, kOutputValue0);
  AddAddNode(subgraph, kIntermediateTensor1, kConstSum, kOutputValue1);
  AddAddNode(subgraph, kIntermediateTensor2, kConstSum, kOutputValue2);
}

void SubgraphBuilder::BuildLargePadSubgraph(Subgraph* subgraph,
                                            const std::vector<int> padding) {
  enum {
    kInputCounter,
    kInputValue0,
    kInputValue1,
    kOutputCounter,
    kOutputValue0,
    kOutputValue1,
    kConstPadding,
    kConstSum,
    kIntermediateTensor0,
    kIntermediateTensor1,
    kTensorCount,
  };

  // kInputCounter -> +-----+
  //                  | ADD | -> kIntermediateTensor0 -> Add -> kOutputCounter
  // kConstsum -----> +-----+
  //
  // kInputValue0 ------> +-----+
  //                      | ADD | -> kIntermediateTensor0(6)
  // kInputValue1   ----> +-----+               |
  //                                            v
  //                                          +-----+
  //                                          | ADD | --> kOutputValue0
  // kConstSum(4)   ----------------------->  +-----+
  //                                             |
  //                                             v
  //                                           +-----+
  //                                           | PAD | --> kOutputValue1
  // kConstStep(4)  ---------------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue0, kInputValue1}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kOutputCounter, kOutputValue0, kOutputValue1}),
      kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstSum, {1}, {1});
  ASSERT_EQ(padding.size() % 2, 0);
  int padding_dims = padding.size();
  CreateConstantTensor(subgraph, kConstPadding, {1, padding_dims}, padding);

  AddAddNode(subgraph, kInputCounter, kConstSum, kIntermediateTensor0);
  AddAddNode(subgraph, kInputCounter, kInputValue1, kIntermediateTensor1);
  AddAddNode(subgraph, kIntermediateTensor0, kConstSum, kOutputCounter);
  AddAddNode(subgraph, kIntermediateTensor1, kConstSum, kOutputValue0);

  int node_index;
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  TfLitePadParams* pad_params =
      reinterpret_cast<TfLitePadParams*>(calloc(1, sizeof(TfLitePadParams)));
  subgraph->AddNodeWithParameters({kOutputValue0, kConstPadding},
                                  {kOutputValue1}, {}, nullptr, 0, pad_params,
                                  pad_reg, &node_index);
}

void SubgraphBuilder::BuildDeepBodySubgraph(Subgraph* subgraph) {
  enum {
    kInputCounter,
    kInputValue,
    kOutputCounter,
    kOutputValue,
    kConstStep,
    kIntermediateTensor,
    kTensorCount,
  };

  // kInputCounter ---> +-----+
  //                    | ADD | -> kOutputCounter
  // kConstStep   ----> +-----+          |
  //                                     v
  //                                   +-----+
  //                                   | ADD |
  // kInputValue  ------------------>  +-----+
  //                                      | kIntermediateTensor
  //                                      v
  //                                    +-----+
  //                                    | ADD | --> kOutputValue
  // kConstStep   ----------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }
  CreateConstantTensor(subgraph, kConstStep, {1}, {1});

  int node_index;
  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  auto* add_reg0 = ops::builtin::Register_ADD();
  add_reg0->builtin_code = kTfLiteBuiltinAdd;
  auto* add_reg1 = ops::builtin::Register_ADD();
  add_reg1->builtin_code = kTfLiteBuiltinAdd;
  auto* add_reg2 = ops::builtin::Register_ADD();
  add_reg2->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({kInputCounter, kConstStep}, {kOutputCounter},
                                  {}, nullptr, 0, params, add_reg0,
                                  &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  subgraph->AddNodeWithParameters({kOutputCounter, kInputValue},
                                  {kIntermediateTensor}, {}, nullptr, 0, params,
                                  add_reg1, &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  subgraph->AddNodeWithParameters({kIntermediateTensor, kConstStep},
                                  {kOutputValue}, {}, nullptr, 0, params,
                                  add_reg2, &node_index);
}

void SubgraphBuilder::BuildAccumulateLoopBodySubgraph(Subgraph* subgraph) {
  const int kInputCounter = 0;
  const int kInputValue = 1;
  const int kOutputCounter = 2;
  const int kOutputValue = 3;
  const int kConstStep = 4;
  const int kTensorCount = 5;

  // kInputCounter(0) --> +-----+
  //                      | ADD | --> kOutputCounter(2)
  // kConstStep(4) -----> +-----+            |
  //                                         |
  //                                         v
  //                                      +-----+
  //                                      | ADD | --> kOutputValue(3)
  // kInputValue(1) ----------------------+-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  SetupTensor(subgraph, kInputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kInputValue, kTfLiteInt32);
  SetupTensor(subgraph, kOutputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kOutputValue, kTfLiteInt32);
  CreateConstantTensor(subgraph, kConstStep, {1}, {1});

  int node_index;
  TfLiteAddParams* params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({0, 4}, {2}, {}, nullptr, 0, params, add_reg,
                                  &node_index);
  params = reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  params->activation = kTfLiteActNone;
  params->pot_scale_int16 = false;
  subgraph->AddNodeWithParameters({2, 1}, {3}, {}, nullptr, 0, params, add_reg,
                                  &node_index);
}

void SubgraphBuilder::BuildPadLoopBodySubgraph(
    Subgraph* subgraph, const std::vector<int>& padding) {
  const int kInputCounter = 0;
  const int kInputValue = 1;
  const int kOutputCounter = 2;
  const int kOutputValue = 3;
  const int kConstStep = 4;
  const int kConstPadding = 5;
  const int kTensorCount = 6;

  // kInputCounter(0) --> +-----+
  //                      | ADD | --> kOutputCounter(2)
  // kConstStep(4) -----> +-----+
  //
  // kInputValue(1) ----> +-----+
  //                      | PAD | --> kOutputValue(3)
  // kConstPadding(5) --> +-----+

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInputCounter, kInputValue}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutputCounter, kOutputValue}), kTfLiteOk);

  SetupTensor(subgraph, kInputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kInputValue, kTfLiteInt32);
  SetupTensor(subgraph, kOutputCounter, kTfLiteInt32);
  SetupTensor(subgraph, kOutputValue, kTfLiteInt32);

  CreateConstantTensor(subgraph, kConstStep, {1}, {1});
  ASSERT_EQ(padding.size() % 2, 0);
  int padding_dims = padding.size();
  CreateConstantTensor(subgraph, kConstPadding, {1, padding_dims}, padding);

  int node_index;
  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;
  auto* add_reg = ops::builtin::Register_ADD();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  subgraph->AddNodeWithParameters({kInputCounter, kConstStep}, {kOutputCounter},
                                  {}, nullptr, 0, add_params, add_reg,
                                  &node_index);
  TfLitePadParams* pad_params =
      reinterpret_cast<TfLitePadParams*>(malloc(sizeof(TfLiteAddParams)));
  auto* pad_reg = ops::builtin::Register_PAD();
  pad_reg->builtin_code = kTfLiteBuiltinPad;
  subgraph->AddNodeWithParameters({kInputValue, kConstPadding}, {kOutputValue},
                                  {}, nullptr, 0, pad_params, pad_reg,
                                  &node_index);
}

void SubgraphBuilder::BuildOutputNotConsumedIfSubgraph(Subgraph* subgraph) {
  enum {
    kInput0,
    kInput1,
    kInput2,
    kInput3,
    kOutput0,
    kOutput1,
    kOutput2,
    kTensorCount
  };

  int num_inputs = 4;
  int num_outputs = 3;
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_tensors[i] = i + num_inputs;
  }
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);
  SetupTensor(subgraph, input_tensors[0], kTfLiteBool);
  for (int i = 1; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteInt32);
  }
  for (int i = 0; i < num_outputs; ++i) {
    SetupTensor(subgraph, output_tensors[i], kTfLiteInt32);
  }

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteIfParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, if_reg, &node_index);
}

void SubgraphBuilder::BuildOutputNotConsumedWhileSubgraph(Subgraph* subgraph) {
  enum {
    kInput0,
    kInput1,
    kInput2,
    kOutput0,
    kOutput1,
    kOutput2,
    kTensorCount
  };

  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput0, kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput0}), kTfLiteOk);

  for (int i = 0; i < kTensorCount; ++i) {
    SetupTensor(subgraph, i, kTfLiteInt32);
  }

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters({0, 1, 2}, {3, 4, 5}, {}, nullptr, 0, params,
                                  while_reg, &node_index);
}

void SubgraphBuilder::BuildFloatIfSubgraph(Subgraph* subgraph, int num_inputs) {
  int num_outputs = num_inputs - 1;
  int first_new_tensor_index;
  ASSERT_EQ(
      subgraph->AddTensors(num_inputs + num_outputs, &first_new_tensor_index),
      kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_tensors[i] = i + num_inputs;
  }
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);

  SetupTensor(subgraph, input_tensors[0], kTfLiteBool);
  for (int i = 1; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteFloat32);
  }
  for (int i = 0; i < num_outputs; ++i) {
    SetupTensor(subgraph, output_tensors[i], kTfLiteFloat32);
  }

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, if_reg, &node_index);
}

void SubgraphBuilder::BuildFloatWhileSubgraph(Subgraph* subgraph,
                                              int num_inputs) {
  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(num_inputs * 2, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
    output_tensors[i] = i + num_inputs;
  }
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);

  for (int i = 0; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteFloat32);
    SetupTensor(subgraph, output_tensors[i], kTfLiteFloat32);
  }

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, while_reg, &node_index);
}

void SubgraphBuilder::BuildMultiInputIfSubgraphWithUnconsumedOutput(
    Subgraph* subgraph, int num_inputs) {
  int num_outputs = num_inputs - 1;
  int first_new_tensor_index;
  ASSERT_EQ(
      subgraph->AddTensors(num_inputs + num_outputs, &first_new_tensor_index),
      kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_tensors[i] = i + num_inputs;
  }
  SetupTensor(subgraph, input_tensors[0], kTfLiteBool);
  for (int i = 1; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteInt32);
  }
  for (int i = 0; i < num_outputs; ++i) {
    SetupTensor(subgraph, output_tensors[i], kTfLiteInt32);
  }

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteIfParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, if_reg, &node_index);

  output_tensors.pop_back();
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);
}

void SubgraphBuilder::BuildMultiInputWhileSubgraphWithUnconsumedOutput(
    Subgraph* subgraph, int num_inputs) {
  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(num_inputs * 2, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
    output_tensors[i] = i + num_inputs;
  }
  for (int i = 0; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteInt32);
    SetupTensor(subgraph, output_tensors[i], kTfLiteInt32);
  }

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, while_reg, &node_index);

  output_tensors.pop_back();
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);
}

void SubgraphBuilder::BuildMultiInputIfSubgraph(Subgraph* subgraph,
                                                int num_inputs) {
  int num_outputs = num_inputs - 1;
  int first_new_tensor_index;
  ASSERT_EQ(
      subgraph->AddTensors(num_inputs + num_outputs, &first_new_tensor_index),
      kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_tensors[i] = i + num_inputs;
  }
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);

  SetupTensor(subgraph, input_tensors[0], kTfLiteBool);
  for (int i = 1; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteInt32);
  }
  for (int i = 0; i < num_outputs; ++i) {
    SetupTensor(subgraph, output_tensors[i], kTfLiteInt32);
  }

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteIfParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, if_reg, &node_index);
}

void SubgraphBuilder::BuildMultiInputWhileSubgraph(Subgraph* subgraph,
                                                   int num_inputs) {
  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(num_inputs * 2, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
    output_tensors[i] = i + num_inputs;
  }
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);

  for (int i = 0; i < num_inputs; ++i) {
    SetupTensor(subgraph, input_tensors[i], kTfLiteInt32);
    SetupTensor(subgraph, output_tensors[i], kTfLiteInt32);
  }

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, while_reg, &node_index);
}

void SubgraphBuilder::BuildWhileSubgraph(Subgraph* subgraph) {
  const int kInput1 = 0;
  const int kInput2 = 1;
  const int kOutput1 = 2;
  const int kOutput2 = 3;
  const int kTensorCount = 4;

  // kInput1(0) --> +-------+ --> kOutput1(2)
  //                | WHILE |
  // kInput2(1) --> +-------+ --> kOutput2(3)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kInput1, kInput2}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput1, kOutput2}), kTfLiteOk);

  SetupTensor(subgraph, kInput1, kTfLiteInt32);
  SetupTensor(subgraph, kInput2, kTfLiteInt32);
  SetupTensor(subgraph, kOutput1, kTfLiteInt32);
  SetupTensor(subgraph, kOutput2, kTfLiteInt32);

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters({0, 1}, {2, 3}, {}, nullptr, 0, params,
                                  while_reg, &node_index);
}

void SubgraphBuilder::BuildAssignRandomValueToVariableSubgraph(
    Subgraph* subgraph) {
  const int kConstResourceId = 0;
  const int kRandomValue = 1;
  const int kTensorCount = 3;

  // Construct a graph like ths:
  //   %1 = random_int()
  //   variable_assign(%0, %1)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({}), kTfLiteOk);

  SetupTensor(subgraph, kRandomValue, kTfLiteInt32);
  CreateConstantTensor(subgraph, kConstResourceId, {1}, {1024});

  int node_index;
  subgraph->AddNodeWithParameters({}, {kRandomValue}, {}, nullptr, 0, nullptr,
                                  ::tflite::ops::custom::Register_RANDOM_INT(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId, kRandomValue}, {}, {}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_ASSIGN_VARIABLE(), &node_index);
}

void SubgraphBuilder::BuildCallOnceAndReadVariableSubgraph(Subgraph* subgraph) {
  const int kConstResourceId = 0;
  const int kOutput = 1;
  const int kTensorCount = 2;

  // Construct a graph like ths:
  //   Output: %1
  //   %1 = read_variable(%0)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kOutput, kTfLiteInt32);
  CreateConstantTensor(subgraph, kConstResourceId, {1}, {1024});

  TfLiteCallOnceParams* params = reinterpret_cast<TfLiteCallOnceParams*>(
      malloc(sizeof(TfLiteCallOnceParams)));
  params->init_subgraph_index = 1;

  int node_index;
  subgraph->AddNodeWithParameters({}, {}, {}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_CALL_ONCE(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId}, {kOutput}, {}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_READ_VARIABLE(), &node_index);
}

void SubgraphBuilder::BuildCallOnceAndReadVariablePlusOneSubgraph(
    Subgraph* subgraph) {
  const int kConstResourceId = 0;
  const int kConstOne = 1;
  const int kReadVariableResult = 2;
  const int kOutput = 3;
  const int kTensorCount = 4;

  // Construct a graph like ths:
  //   Output: %3
  //   %2 = read_variable(%0)
  //   %3 = add(%2, %1)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetInputs({}), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kReadVariableResult, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteInt32);
  CreateConstantTensor(subgraph, kConstResourceId, {1}, {1024});
  CreateConstantTensor(subgraph, kConstOne, {1}, {1});

  TfLiteCallOnceParams* params = reinterpret_cast<TfLiteCallOnceParams*>(
      malloc(sizeof(TfLiteCallOnceParams)));
  params->init_subgraph_index = 1;

  int node_index;
  subgraph->AddNodeWithParameters({}, {}, {}, nullptr, 0, params,
                                  ::tflite::ops::builtin::Register_CALL_ONCE(),
                                  &node_index);
  subgraph->AddNodeWithParameters(
      {kConstResourceId}, {kReadVariableResult}, {}, nullptr, 0, nullptr,
      ::tflite::ops::builtin::Register_READ_VARIABLE(), &node_index);

  TfLiteAddParams* add_params =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  add_params->activation = kTfLiteActNone;
  subgraph->AddNodeWithParameters(
      {kReadVariableResult, kConstOne}, {kOutput}, {}, nullptr, 0, add_params,
      ::tflite::ops::builtin::Register_ADD(), &node_index);
}

void SubgraphBuilder::BuildLessEqualCondSubgraphWithDynamicTensor(
    Subgraph* subgraph, int rhs) {
  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kOutput = 3;
  const int kConstRhs = 4;
  const int kTensorCount = 5;

  // kIntegerInput(2) --> +------------+
  //                      | LESS_EQUAL | --> kOutput(3)
  //     kConstRhs(4) --> +------------+
  //
  // kStringInput1(0) --> (unused)
  // kStringInput2(1) --> (unused)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs({kOutput}), kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kOutput, kTfLiteBool);

  auto* le_reg = ops::builtin::Register_LESS_EQUAL();
  le_reg->builtin_code = kTfLiteBuiltinLessEqual;

  CreateConstantTensor(subgraph, kConstRhs, {1}, {rhs});
  int node_index;
  subgraph->AddNodeWithParameters({kIntegerInput, kConstRhs}, {kOutput}, {},
                                  nullptr, 0, nullptr, le_reg, &node_index);
}

void SubgraphBuilder::BuildBodySubgraphWithDynamicTensor(Subgraph* subgraph) {
  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kStringOutput1 = 0;  // Forwarded of the `kStringInput1` tensor.
  const int kStringOutput2 = 4;
  const int kIntegerOutput = 5;
  const int kConst = 6;
  const int kTensorCount = 7;

  // Construct a graph like this:
  //   %5 = tf.Add(%2, 1)
  //   %4 = tf.Fill(%0, %5)
  //   yield(%0, %4, %5)

  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kStringOutput1, kStringOutput2, kIntegerOutput}),
      kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);
  SetupTensor(subgraph, kConst, kTfLiteInt32);

  CreateConstantTensor(subgraph, kConst, {1}, {1});
  AddAddNode(subgraph, kIntegerInput, kConst, kIntegerOutput);

  int node_index;
  auto* fill_reg = ops::builtin::Register_FILL();
  fill_reg->builtin_code = kTfLiteBuiltinFill;
  subgraph->AddNodeWithParameters({kIntegerOutput, kStringInput1},
                                  {kStringOutput2}, {}, nullptr, 0, nullptr,
                                  fill_reg, &node_index);
}

void SubgraphBuilder::BuildIfSubgraphWithDynamicTensor(Subgraph* subgraph) {
  enum {
    kBoolInput0,
    kStringInput1,
    kStringInput2,
    kIntegerInput,
    kStringOutput1,
    kStringOutput2,
    kIntegerOutput,
    kTensorCount
  };

  int num_inputs = 4;
  int num_outputs = num_inputs - 1;
  // Create a if op with 2 string tensor and 1 integer tensor.
  int first_new_tensor_index;
  std::vector<int> input_tensors(num_inputs);
  std::vector<int> output_tensors(num_outputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = i;
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_tensors[i] = i + num_inputs;
  }
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs(input_tensors), kTfLiteOk);
  ASSERT_EQ(subgraph->SetOutputs(output_tensors), kTfLiteOk);

  SetupTensor(subgraph, kBoolInput0, kTfLiteBool);
  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);

  TfLiteIfParams* params =
      reinterpret_cast<TfLiteIfParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->then_subgraph_index = 1;
  params->else_subgraph_index = 2;
  auto* if_reg = ops::builtin::Register_IF();
  if_reg->builtin_code = kTfLiteBuiltinIf;

  int node_index;
  subgraph->AddNodeWithParameters(input_tensors, output_tensors, {}, nullptr, 0,
                                  params, if_reg, &node_index);
}

void SubgraphBuilder::BuildWhileSubgraphWithDynamicTensor(Subgraph* subgraph) {
  const int kStringInput1 = 0;
  const int kStringInput2 = 1;
  const int kIntegerInput = 2;
  const int kStringOutput1 = 3;
  const int kStringOutput2 = 4;
  const int kIntegerOutput = 5;
  const int kTensorCount = 6;

  // Create a while op with 2 string tensor and 1 integer tensor.
  int first_new_tensor_index;
  ASSERT_EQ(subgraph->AddTensors(kTensorCount, &first_new_tensor_index),
            kTfLiteOk);
  ASSERT_EQ(first_new_tensor_index, 0);
  ASSERT_EQ(subgraph->SetInputs({kStringInput1, kStringInput2, kIntegerInput}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph->SetOutputs({kStringOutput1, kStringOutput2, kIntegerOutput}),
      kTfLiteOk);

  SetupTensor(subgraph, kStringInput1, kTfLiteString);
  SetupTensor(subgraph, kStringInput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerInput, kTfLiteInt32);
  SetupTensor(subgraph, kStringOutput1, kTfLiteString);
  SetupTensor(subgraph, kStringOutput2, kTfLiteString);
  SetupTensor(subgraph, kIntegerOutput, kTfLiteInt32);

  TfLiteWhileParams* params =
      reinterpret_cast<TfLiteWhileParams*>(malloc(sizeof(TfLiteWhileParams)));
  params->cond_subgraph_index = 1;
  params->body_subgraph_index = 2;
  auto* while_reg = ops::builtin::Register_WHILE();
  while_reg->builtin_code = kTfLiteBuiltinWhile;

  int node_index;
  subgraph->AddNodeWithParameters(
      {kStringInput1, kStringInput2, kIntegerInput},
      {kStringOutput1, kStringOutput2, kIntegerOutput}, {}, nullptr, 0, params,
      while_reg, &node_index);
}

void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data) {
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    tensor->data.i32[i] = data[i];
  }
}

void FillScalarStringTensor(TfLiteTensor* tensor, const std::string& data) {
  StringRef str_ref;
  str_ref.str = data.c_str();
  str_ref.len = data.size();
  DynamicBuffer buf;
  buf.AddString(str_ref);
  buf.WriteToTensor(tensor, /*new_shape=*/TfLiteIntArrayCreate(0));
}

void CheckScalarStringTensor(const TfLiteTensor* tensor,
                             const std::string& data) {
  ASSERT_EQ(tensor->dims->size, 0);
  ASSERT_EQ(tensor->type, kTfLiteString);
  StringRef str_ref = GetString(tensor, 0);
  EXPECT_EQ(std::string(str_ref.str, str_ref.len), data);
}

void CheckStringTensor(const TfLiteTensor* tensor,
                       const std::vector<int>& shape,
                       const std::vector<std::string>& data) {
  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteString);
  int count = GetStringCount(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    StringRef str_ref = GetString(tensor, i);
    EXPECT_EQ(std::string(str_ref.str, str_ref.len), data[i]);
  }
}
void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data) {
  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteInt32);
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(tensor->data.i32[i], data[i]);
  }
}

void CheckBoolTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                     const std::vector<bool>& data) {
  ASSERT_EQ(tensor->dims->size, shape.size());
  for (int i = 0; i < tensor->dims->size; ++i) {
    ASSERT_EQ(tensor->dims->data[i], shape[i]);
  }
  ASSERT_EQ(tensor->type, kTfLiteBool);
  int count = NumElements(tensor);
  ASSERT_EQ(count, data.size());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(tensor->data.b[i], data[i]);
  }
}

}  // namespace subgraph_test_util
}  // namespace tflite
