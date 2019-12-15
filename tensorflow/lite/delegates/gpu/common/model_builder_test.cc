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

#include "tensorflow/lite/delegates/gpu/common/model_builder.h"

#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace gpu {
namespace {

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank0) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.name = "tensor_name";
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 4;
  TensorRef<BHWC> tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::FLOAT32);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 1, 1, 1));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank1) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.name = "tensor_name";
  tflite_tensor.type = TfLiteType::kTfLiteInt32;
  tflite_tensor.dims = TfLiteIntArrayCreate(2);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.dims->data[1] = 5;
  TensorRef<BHWC> tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::INT32);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 1, 1, 5));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank2) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.name = "tensor_name";
  tflite_tensor.type = TfLiteType::kTfLiteInt64;
  tflite_tensor.dims = TfLiteIntArrayCreate(3);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.dims->data[1] = 5;
  tflite_tensor.dims->data[2] = 6;
  TensorRef<BHWC> tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::INT64);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 1, 5, 6));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank3) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.name = "tensor_name";
  tflite_tensor.type = TfLiteType::kTfLiteUInt8;
  tflite_tensor.dims = TfLiteIntArrayCreate(4);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.dims->data[1] = 5;
  tflite_tensor.dims->data[2] = 6;
  tflite_tensor.dims->data[3] = 7;
  TensorRef<BHWC> tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::UINT8);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 5, 6, 7));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefFailsForRankLT0) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.name = "tensor_name";
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(0);
  TensorRef<BHWC> tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  // TODO(b/130054481): Cover scalar.
  EXPECT_FALSE(status.ok());
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefFailsForRankGT3) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.name = "tensor_name";
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(5);
  TensorRef<BHWC> tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  EXPECT_FALSE(status.ok());
}

class InterpreterFp16 {
 public:
  explicit InterpreterFp16(TfLiteBuiltinOperator op) {
    void* builtin_data = malloc(sizeof(int));
    EXPECT_EQ(interpreter_.AddTensors(5), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 1}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({4}), kTfLiteOk);

    // Add a Dequantize Node.
    const TfLiteRegistration reg_dequant0 = {
        nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant0),
              kTfLiteOk);

    // Add a Dequantize Node.
    const TfLiteRegistration reg_dequant1 = {
        nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{2}, /*outputs=*/{3}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant1),
              kTfLiteOk);

    // Add a node that GPU delegate can parse.
    const TfLiteRegistration reg_op0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        op};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{1, 3}, /*outputs=*/{4}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_op0),
              kTfLiteOk);

    // Set inputs to Dequantize node to the fp16 type, and outputs
    // to fp32 type.
    const std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            0, TfLiteType::kTfLiteFloat16, "t0", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat16, "t2", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat32, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);

    exec_plan_ = TfLiteIntArrayCreate(3);
    exec_plan_->data[0] = 0;
    exec_plan_->data[1] = 1;
    exec_plan_->data[2] = 2;
  }

  ~InterpreterFp16() { TfLiteIntArrayFree(exec_plan_); }

  Subgraph* GetSubgraph() { return interpreter_.subgraph(0); }
  TfLiteIntArray* exec_plan() const { return exec_plan_; }

 private:
  Interpreter interpreter_;
  TfLiteIntArray* exec_plan_;
};

InterpreterFp16* interpreter_fp16_add_op =
    new InterpreterFp16(kTfLiteBuiltinAdd);

TEST(ModelBuilderTest, GetOpsToReplacePrunesFp16DequantizeNodes) {
  // Before pruning, the graph has three nodes:
  //
  //   t0 (FP16) -> DequantNode -> t1 (FP32) -> Add -> t4
  //   t2 (FP16) -> DequantNode -> t3 (FP32) --/
  //
  // OpsToReplace should choose all three nodes for replacement, and
  // the graph on the GPU will look like this (no Dequants):
  //
  //   t0 (FP16) --> Add -> t4
  //   t2 (FP16) --/
  //
  TfLiteContext* context = interpreter_fp16_add_op->GetSubgraph()->context();
  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_add_op->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    auto& node_and_reg = interpreter_fp16_add_op->GetSubgraph()
                             ->nodes_and_registration()[node_index];
    *node = &node_and_reg.first;
    *registration = &node_and_reg.second;
    return kTfLiteOk;
  };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // Replace all nodes.
  EXPECT_EQ(ops_to_replace->size, 3);
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, ops_to_replace->data[2], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

// This interpreter instance is created at global scope to test *exactly*
// the GetOpsToReplace function alone, and not the sequence of function calls
// that includes GetOpsToReplace when calling ModifyGraphWithDelegate.
// A TfLiteContext is needed to test GetOpsToReplace, but TfLiteContexts
// intentionally make it difficult to call certain functions in a
// non-delegate context (see tensorflow/lite/subgraph/subgraph.cc for details)
// We create our own GetExecutionPlan and GetNodeAndRegistration lambdas
// inside each test, but we can't use local captures without changing the
// function signature. Therefore, this test data lives at global scope
// in order to be accessible inside the lambda.

InterpreterFp16* interpreter_fp16_gt_op =
    new InterpreterFp16(kTfLiteBuiltinGreater);

TEST(ModelBuilderTest, GetOpsToReplaceKeepsFp16DequantizeNodes) {
  // Before pruning, the graph has three nodes:
  //
  //   t0 (FP16) -> DequantNode -> t1 (FP32) -> Greater Op -> t4
  //   t2 (FP16) -> DequantNode -> t3 (FP32) --/
  //
  // Because there is no GPU equivalent for the Greater op, we don't prune
  // the Dequantize nodes.

  TfLiteContext* context = interpreter_fp16_gt_op->GetSubgraph()->context();
  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_gt_op->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    auto& node_and_reg = interpreter_fp16_gt_op->GetSubgraph()
                             ->nodes_and_registration()[node_index];
    *node = &node_and_reg.first;
    *registration = &node_and_reg.second;
    return kTfLiteOk;
  };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // No nodes were found to replace.
  EXPECT_EQ(ops_to_replace->size, 0);
  // Inputs to Greater op are still fp32.
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  const int kGreaterOpIndex = 2;
  context->GetNodeAndRegistration(context, kGreaterOpIndex, &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat32);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat32);
  TfLiteIntArrayFree(ops_to_replace);
}

class InterpreterFp32 {
 public:
  InterpreterFp32() {
    void* builtin_data = malloc(sizeof(int));
    EXPECT_EQ(interpreter_.AddTensors(4), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 2}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({3}), kTfLiteOk);

    // Add a Dequantize Node with uint8 input.
    const TfLiteRegistration reg_dequant0 = {/*init=*/nullptr,
                                             /*free=*/nullptr,
                                             /*prepare=*/nullptr,
                                             /*invoke=*/nullptr,
                                             /*profiling_string=*/nullptr,
                                             kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant0),
              kTfLiteOk);

    // Add a node that GPU delegate can parse.
    const TfLiteRegistration reg_add0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{1, 2}, /*outputs=*/{3}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add0),
              kTfLiteOk);

    const std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(interpreter_.SetTensorParametersReadWrite(
                  0, TfLiteType::kTfLiteUInt8, "t0", dims, quantization, false),
              kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat32, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat32, "t2", dims, quantization, false),
        kTfLiteOk);
    exec_plan_ = TfLiteIntArrayCreate(2);
    exec_plan_->data[0] = 0;
    exec_plan_->data[1] = 1;
  }

  ~InterpreterFp32() { TfLiteIntArrayFree(exec_plan_); }

  Subgraph* GetSubgraph() { return interpreter_.subgraph(0); }
  TfLiteIntArray* exec_plan() const { return exec_plan_; }

 private:
  Interpreter interpreter_;
  TfLiteIntArray* exec_plan_;
};

InterpreterFp32* interpreter_fp32 = new InterpreterFp32();

TEST(ModelBuilderTest, GetOpsToReplaceDoesNotPruneUint8) {
  // A graph with a Dequant node with uint8 input
  // is not pruned. The delegate will attempt to replace it
  // with a GPU op, but this op is currently not supported on
  // the GPU. Therefore, the Dequant op and all downstream ops
  // will be scheduled to run on the CPU.
  //
  //   t0 (uint8) --> Dequant --> t1 (FP32) --> Add -> t3
  //                              t2 (FP32) --/
  //
  TfLiteContext* context = interpreter_fp32->GetSubgraph()->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp32->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    auto& node_and_reg =
        interpreter_fp32->GetSubgraph()->nodes_and_registration()[node_index];
    *node = &node_and_reg.first;
    *registration = &node_and_reg.second;
    return kTfLiteOk;
  };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // No ops are run on the GPU, since the Dequant op is not pruned and must run
  // on the CPU.
  EXPECT_EQ(ops_to_replace->size, 0);
  TfLiteIntArrayFree(ops_to_replace);
}

class InterpreterMultiNode {
 public:
  InterpreterMultiNode() {
    void* builtin_data = malloc(sizeof(int));
    EXPECT_EQ(interpreter_.AddTensors(8), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 1, 2}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({6, 7}), kTfLiteOk);

    // Add 3 Dequantize Nodes with float16 input.
    for (int i = 0; i < 3; ++i) {
      const TfLiteRegistration reg_dequant = {/*init=*/nullptr,
                                              /*free=*/nullptr,
                                              /*prepare=*/nullptr,
                                              /*invoke=*/nullptr,
                                              /*profiling_string=*/nullptr,
                                              kTfLiteBuiltinDequantize};
      EXPECT_EQ(interpreter_.AddNodeWithParameters(
                    /*inputs=*/{i}, /*outputs=*/{i + 3}, /*init_data=*/nullptr,
                    /*init_data_size=*/0, /*builtin_data=*/nullptr,
                    /*registration=*/&reg_dequant),
                kTfLiteOk);
    }

    // Add the ADD op node that GPU delegate supports.
    const TfLiteRegistration reg_add0 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};

    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{4, 5}, /*outputs=*/{7}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add0),
              kTfLiteOk);

    // Add the GreaterThan op node that GPU delegate doesn't support.
    const TfLiteRegistration reg_greater = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int(1));
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinGreater};

    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{3, 4}, /*outputs=*/{6}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_greater),
              kTfLiteOk);

    const std::vector<int> dims = {1};
    TfLiteQuantization quantization;
    quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            0, TfLiteType::kTfLiteFloat16, "t0", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat16, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteFloat16, "t2", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteFloat32, "t4", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            5, TfLiteType::kTfLiteFloat32, "t5", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            6, TfLiteType::kTfLiteFloat32, "t5", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            7, TfLiteType::kTfLiteFloat32, "t5", dims, quantization, false),
        kTfLiteOk);
    exec_plan_ = TfLiteIntArrayCreate(5);
    exec_plan_->data[0] = 0;
    exec_plan_->data[1] = 1;
    exec_plan_->data[2] = 2;
    exec_plan_->data[3] = 3;
    exec_plan_->data[4] = 4;
  }

  ~InterpreterMultiNode() { TfLiteIntArrayFree(exec_plan_); }

  Subgraph* GetSubgraph() { return interpreter_.subgraph(0); }
  TfLiteIntArray* exec_plan() const { return exec_plan_; }

 private:
  Interpreter interpreter_;
  TfLiteIntArray* exec_plan_;
};

InterpreterMultiNode* interpreter_mn = new InterpreterMultiNode();

TEST(ModelBuilderTest, GetOpsToReplaceSelectsCorrectDequants) {
  // A graph with three Dequant nodes feeding two ops, 'Add' and 'Greater'.
  // 'Add' can be replaced by the GPU delegate, but 'Greater' can not.
  //   t0 (FP16) --> Dequant --> t3 (FP32) --> Greater -> t6
  //   t1 (FP16) --> Dequant --> t4 (FP32) --/
  //                                       --\
  //   t3 (FP16) --> Dequant --> t5 (FP32) --> Add -> t7
  //
  //  OpsToReplace should replace the 'Add' op and the Dequant outputing
  //  t5, but leave the other Dequant nodes because 'Greater' must run
  //  on the CPU.
  TfLiteContext* context = interpreter_mn->GetSubgraph()->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_mn->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    auto& node_and_reg =
        interpreter_mn->GetSubgraph()->nodes_and_registration()[node_index];
    *node = &node_and_reg.first;
    *registration = &node_and_reg.second;
    return kTfLiteOk;
  };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  EXPECT_EQ(ops_to_replace->size, 2);
  // Op at index 2 is the Dequant op (t3 -> t5).
  EXPECT_EQ(ops_to_replace->data[0], 2);
  // Op at index 3 is the Add op.
  EXPECT_EQ(ops_to_replace->data[1], 3);

  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  // Verify that Add op has fp16 inputs.
  context->GetNodeAndRegistration(context, ops_to_replace->data[1], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
