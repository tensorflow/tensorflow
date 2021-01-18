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

#include <stddef.h>
#include <stdint.h>

#include <cstdlib>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/interpreter.h"

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

class DelegatedInterpreter {
 public:
  explicit DelegatedInterpreter(int num_nodes) {
    exec_plan_ = TfLiteIntArrayCreate(num_nodes);
  }
  virtual ~DelegatedInterpreter() {
    TfLiteIntArrayFree(exec_plan_);
    for (auto params : delegate_params_) {
      TfLiteIntArrayFree(params.nodes_to_replace);
      TfLiteIntArrayFree(params.input_tensors);
      TfLiteIntArrayFree(params.output_tensors);
    }
  }

  // Get the TfLiteContext to be mocked for swapping out functions that have to
  // be called inside delegate (i.e. in delegate kernel mode).
  TfLiteContext* context() { return interpreter_.primary_subgraph().context(); }

  // node(int) and registration(int) are used to implement
  // GetNodeAndRegistration.  We can't implement those using
  //   TfLiteContext *context = interpreter_.primary_subgraph().context();
  //   context->GetNodeAndRegistration(context, &node, &registration);
  // here, because calling GetNodeAndRegistration from within it's own
  // implementation would lead to an infinite loop.
  // Instead, we just call node_and_registration and use a const_cast.
  // These const_casts are a bit ugly, but I think less ugly than exposing
  // the private GetNodeAndRegistration method in Subgraph as public,
  // or making this class a friend of Subgraph.
  TfLiteNode* node(int index) {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration =
        interpreter_.primary_subgraph().node_and_registration(index);
    return const_cast<TfLiteNode*>(&node_and_registration->first);
  }
  TfLiteRegistration* registration(int index) {
    const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration =
        interpreter_.primary_subgraph().node_and_registration(index);
    return const_cast<TfLiteRegistration*>(&node_and_registration->second);
  }

  TfLiteIntArray* exec_plan() const { return exec_plan_; }
  TfLiteDelegateParams* add_delegate_params() {
    delegate_params_.push_back(TfLiteDelegateParams());
    return &delegate_params_.back();
  }
  TfLiteDelegateParams* delegate_params() { return &delegate_params_.front(); }
  int num_delegate_params() { return delegate_params_.size(); }

 protected:
  Interpreter interpreter_;

 private:
  // The manually-set execution plan for this delegated interpreter.
  TfLiteIntArray* exec_plan_;

  // The TfLiteDelegateParams object that's manually populated inside the mocked
  // TfLiteContext::PreviewDelegatePartitioning.
  std::vector<TfLiteDelegateParams> delegate_params_;
};

class InterpreterFp16 : public DelegatedInterpreter {
 public:
  explicit InterpreterFp16(TfLiteBuiltinOperator op,
                           bool const_dequantize_inputs = true)
      : DelegatedInterpreter(3) {
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
    if (const_dequantize_inputs) {
      // This simulates the dequantize inputs being constants in the graph.
      // If this is not true, FP16GraphPartitionHelper should not consider the
      // corresponding DEQUANTIZE ops.
      auto* tensor0 = interpreter_.tensor(0);
      auto* tensor2 = interpreter_.tensor(2);
      tensor0->allocation_type = kTfLiteMmapRo;
      tensor2->allocation_type = kTfLiteMmapRo;
    }
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteFloat32, "t1", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
  }
};

// **NOTE**: we have several interpreter instances created at global scope to
// test *exactly* the GetOpsToReplace function alone, and not the sequence of
// function calls that includes GetOpsToReplace when calling
// ModifyGraphWithDelegate. A TfLiteContext is needed to test GetOpsToReplace,
// but TfLiteContexts intentionally make it difficult to call certain functions
// in a non-delegate context (see tensorflow/lite/subgraph/subgraph.cc for
// details) We create our own GetExecutionPlan, GetNodeAndRegistration and
// PreviewDelegatePartitioning lambdas inside each test, but we can't use local
// captures without changing the function signature. Therefore, this test data
// lives at global scope in order to be accessible inside the lambda.

InterpreterFp16* interpreter_fp16_add_op =
    new InterpreterFp16(kTfLiteBuiltinAdd);

TEST(ModelBuilderTest, GetOpsToReplaceAcceptsFp16DequantizeNodes) {
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
  TfLiteContext* context = interpreter_fp16_add_op->context();

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
    *node = interpreter_fp16_add_op->node(node_index);
    *registration = interpreter_fp16_add_op->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The partitioner should accept only the Add op initially.
        EXPECT_EQ(nodes_to_replace->size, 1);
        // Single partition output.
        auto params = interpreter_fp16_add_op->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 2;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 4;

        *partition_params_array = interpreter_fp16_add_op->delegate_params();
        *num_partitions = interpreter_fp16_add_op->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // The Dequant nodes are added to ops_to_replace as a post-processing step by
  // the FP16GraphPartitioner. ADD is delegated with its inputs pointing to the
  // FP16 inputs.
  EXPECT_EQ(ops_to_replace->size, 3);
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, ops_to_replace->data[0], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterFp16* interpreter_fp16_non_constant =
    new InterpreterFp16(kTfLiteBuiltinAdd, /*const_dequantize_inputs=*/false);

// Same as GetOpsToReplaceAcceptsFp16DequantizeNodes, but the DEQUANTIZE inputs
// are not constant. As a result, we don't allow the delegate to accept them.
TEST(ModelBuilderTest, GetOpsToReplaceRejectsNonConstantFp16DequantizeNodes) {
  TfLiteContext* context = interpreter_fp16_non_constant->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_fp16_non_constant->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_fp16_non_constant->node(node_index);
    *registration = interpreter_fp16_non_constant->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The partitioner should accept only the Add op initially.
        EXPECT_EQ(nodes_to_replace->size, 1);
        // Single partition output.
        auto params = interpreter_fp16_non_constant->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 2;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 4;

        *partition_params_array =
            interpreter_fp16_non_constant->delegate_params();
        *num_partitions = interpreter_fp16_non_constant->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // Only ADD is delegated, with FP32 (dequantized) inputs.
  EXPECT_EQ(ops_to_replace->size, 1);
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, ops_to_replace->data[0], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat32);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat32);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterFp16* interpreter_fp16_gt_op =
    new InterpreterFp16(kTfLiteBuiltinGreater);

TEST(ModelBuilderTest, GetOpsToReplaceRejectsFp16DequantizeNodes) {
  // Before pruning, the graph has three nodes:
  //
  //   t0 (FP16) -> DequantNode -> t1 (FP32) -> Greater Op -> t4
  //   t2 (FP16) -> DequantNode -> t3 (FP32) --/
  //
  // Because there is no GPU equivalent for the Greater op, we don't choose any
  // nodes.

  TfLiteContext* context = interpreter_fp16_gt_op->context();
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
    *node = interpreter_fp16_gt_op->node(node_index);
    *registration = interpreter_fp16_gt_op->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // No selected nodes.
        EXPECT_EQ(nodes_to_replace->size, 0);
        *partition_params_array = nullptr;
        *num_partitions = 0;
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

class InterpreterFp32 : public DelegatedInterpreter {
 public:
  InterpreterFp32() : DelegatedInterpreter(2) {
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

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
  }
};

InterpreterFp32* interpreter_fp32 = new InterpreterFp32();

TEST(ModelBuilderTest, GetOpsToReplaceDoesNotPruneUint8) {
  // A graph with a Dequant node with uint8 input is not pruned. As this op is
  // currently not supported on the GPU. Therefore, the Dequant op will be
  // scheduled to run on the CPU while the remaining supported op Add on the
  // GPU.
  //
  //   t0 (uint8) --> Dequant --> t1 (FP32) --> Add -> t3
  //                              t2 (FP32) --/
  //
  TfLiteContext* context = interpreter_fp32->context();

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
    *node = interpreter_fp32->node(node_index);
    *registration = interpreter_fp32->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        auto params = interpreter_fp32->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 1;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 2;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 3;

        *partition_params_array = interpreter_fp32->delegate_params();
        *num_partitions = interpreter_fp32->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // As the Dequant op is not pruned and the ADD op could run on GPU, we have
  // 1 partition.
  EXPECT_EQ(ops_to_replace->size, 1);
  // ADD at index 1.
  EXPECT_EQ(1, ops_to_replace->data[0]);

  TfLiteIntArrayFree(ops_to_replace);
}

class Interpreter2Fp32 : public DelegatedInterpreter {
 public:
  Interpreter2Fp32() : DelegatedInterpreter(4) {
    void* builtin_data = malloc(sizeof(int));
    EXPECT_EQ(interpreter_.AddTensors(8), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 2, 4, 6}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({7}), kTfLiteOk);

    // Add a Dequantize Node with uint8 input.
    const TfLiteRegistration reg_dequant = {/*init=*/nullptr,
                                            /*free=*/nullptr,
                                            /*prepare=*/nullptr,
                                            /*invoke=*/nullptr,
                                            /*profiling_string=*/nullptr,
                                            kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant),
              kTfLiteOk);

    // Add an ADD node that GPU delegate can parse.
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

    // Add a Pack Node that GPU delegate doesn't support
    const TfLiteRegistration reg_pack = {/*init=*/nullptr,
                                         /*free=*/nullptr,
                                         /*prepare=*/nullptr,
                                         /*invoke=*/nullptr,
                                         /*profiling_string=*/nullptr,
                                         kTfLiteBuiltinPack};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{3, 4}, /*outputs=*/{5}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_pack),
              kTfLiteOk);

    const TfLiteRegistration reg_add1 = {
        [](TfLiteContext* context, const char* buffer, size_t length) {
          return reinterpret_cast<void*>(new int[2]);
        },
        [](TfLiteContext* context, void* buffer) {
          delete reinterpret_cast<int*>(buffer);
        },
        nullptr,
        nullptr,
        nullptr,
        kTfLiteBuiltinAdd};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{5, 6}, /*outputs=*/{7}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add1),
              kTfLiteOk);

    std::vector<int> dims = {1};
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
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteFloat32, "t4", dims, quantization, false),
        kTfLiteOk);

    dims.push_back(2);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            5, TfLiteType::kTfLiteFloat32, "t5", dims, quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            6, TfLiteType::kTfLiteFloat32, "t6", dims, quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
    exec_plan()->data[3] = 3;
  }
};

Interpreter2Fp32* interpreter2_fp32 = new Interpreter2Fp32();

TEST(ModelBuilderTest, GetOpsToReplaceMultiplePartitions) {
  // A graph with a Dequant node with uint8 input, a Pack node are not pruned.
  // As these ops are currently not supported on the GPU, they will be scheduled
  // to run on the CPU while the remaining supported op Add on the GPU.
  //
  //   t0 (uint8) -> Dequant(0) -> t1 (FP32) -> Add(1) -> t3 (FP32) -> PACK (2)
  //                               t2 (FP32) -/           t4 (FP32) -/
  //   PACK (2) -> t5 (FP32) -> Add(3) -> t7
  //            -> t6 (FP32) -/
  //
  TfLiteContext* context = interpreter2_fp32->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter2_fp32->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter2_fp32->node(node_index);
    *registration = interpreter2_fp32->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        auto params = interpreter2_fp32->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 1;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 2;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 3;

        params = interpreter2_fp32->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 3;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 5;
        params->input_tensors->data[1] = 6;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 7;

        *partition_params_array = interpreter2_fp32->delegate_params();
        *num_partitions = interpreter2_fp32->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops=*/false, /*max_delegated_partitions*/ 2);

  // As the Dequant op is not pruned and the ADD op could run on GPU, we have
  // 2 partitions.
  EXPECT_EQ(ops_to_replace->size, 2);
  // ADD at index 1.
  EXPECT_EQ(1, ops_to_replace->data[0]);
  // ADD at index 3.
  EXPECT_EQ(3, ops_to_replace->data[1]);

  TfLiteIntArrayFree(ops_to_replace);
}

class InterpreterMultiNode : public DelegatedInterpreter {
 public:
  explicit InterpreterMultiNode(bool both_ops_supported = true)
      : DelegatedInterpreter(5) {
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

    if (both_ops_supported) {
      // Add 2 ADD ops.
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

      const TfLiteRegistration reg_add1 = {
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
                    /*inputs=*/{3, 4}, /*outputs=*/{6}, /*init_data=*/nullptr,
                    /*init_data_size=*/0,
                    /*builtin_data=*/builtin_data,
                    /*registration=*/&reg_add1),
                kTfLiteOk);
    } else {
      // Add the GREATER op node that GPU delegate doesn't support.
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
    }
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
    // Simulate DEQUANTIZE inputs being constants.
    auto* tensor0 = interpreter_.tensor(0);
    auto* tensor1 = interpreter_.tensor(1);
    auto* tensor2 = interpreter_.tensor(2);
    tensor0->allocation_type = kTfLiteMmapRo;
    tensor1->allocation_type = kTfLiteMmapRo;
    tensor2->allocation_type = kTfLiteMmapRo;
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

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
    exec_plan()->data[3] = 3;
    exec_plan()->data[4] = 4;
  }
};

InterpreterMultiNode* interpreter_mn =
    new InterpreterMultiNode(/*both_ops_supported*/ false);

TEST(ModelBuilderTest, GetOpsToReplaceSelectsCorrectFp16Nodes_SinglePartition) {
  // A graph with three Dequant nodes feeding two ops, 'Add' and 'Greater'.
  // 'Add' can be replaced by the GPU delegate, but 'Greater' can not.
  //   t0 (FP16) --> Dequant(0) --> t3 (FP32) --> Greater(3) -> t6
  //   t1 (FP16) --> Dequant(1) --> t4 (FP32) --/
  //                                          --\
  //   t3 (FP16) --> Dequant(2) --> t5 (FP32) --> Add(4) -> t7
  //
  //  OpsToReplace should accept 'Add' & the Dequant nodes that only output to
  //  it (in this case, Dequant(2)).
  TfLiteContext* context = interpreter_mn->context();

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
    *node = interpreter_mn->node(node_index);
    *registration = interpreter_mn->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The FP16GraphPartitioner should only mark the ADD op as accepted.
        EXPECT_EQ(nodes_to_replace->size, 1);
        EXPECT_EQ(nodes_to_replace->data[0], 4);
        // Single partition.
        auto params = interpreter_mn->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 4;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 1;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 7;

        *partition_params_array = interpreter_mn->delegate_params();
        *num_partitions = interpreter_mn->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);

  // Post-PreviewDelegatePartitioning, the partitioner will add Dequant(2) to
  // ops_to_replace, since it only outputs to a delegated node.
  EXPECT_EQ(ops_to_replace->size, 2);
  // Op at index 4 is the Add op.
  EXPECT_EQ(ops_to_replace->data[0], 4);
  EXPECT_EQ(ops_to_replace->data[1], 2);
  // Verify that Add op has fp16 inputs.
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  context->GetNodeAndRegistration(context, ops_to_replace->data[0], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  TfLiteIntArrayFree(ops_to_replace);
}

InterpreterMultiNode* interpreter_mn2 =
    new InterpreterMultiNode(/*both_ops_supported*/ true);
TEST(ModelBuilderTest,
     GetOpsToReplaceSelectsCorrectFp16Nodes_MultiplePartitions) {
  // A graph with three Dequant nodes feeding two Add ops.
  //   t0 (FP16) --> Dequant(0) --> t3 (FP32) --> Add(3) -> t6
  //   t1 (FP16) --> Dequant(1) --> t4 (FP32) --/
  //                                          --\
  //   t3 (FP16) --> Dequant(2) --> t5 (FP32) --> Add(4) -> t7
  //
  // In this test case, we purposely partition Add(3) & Add(4) into different
  // partitions, to check if Dequant nodes that output *only* to the first
  // partition nodes are accepted.

  TfLiteContext* context = interpreter_mn2->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_mn2->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_mn2->node(node_index);
    *registration = interpreter_mn2->registration(node_index);
    return kTfLiteOk;
  };

  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        // The FP16GraphPartitioner should only mark both ADD ops as accepted.
        EXPECT_EQ(nodes_to_replace->size, 2);
        EXPECT_EQ(nodes_to_replace->data[0], 3);
        EXPECT_EQ(nodes_to_replace->data[1], 4);
        // Technically, both ADD ops should end up in the same partition.
        // But we put them in different partitions to test post-processing with
        // DEQUANTIZE nodes.
        // First partition with Add(3).
        auto params = interpreter_mn2->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 3;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 3;
        params->input_tensors->data[1] = 4;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 6;
        // Second partition with Add(4).
        params = interpreter_mn2->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(1);
        params->nodes_to_replace->data[0] = 4;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 4;
        params->input_tensors->data[1] = 5;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 7;

        *partition_params_array = interpreter_mn2->delegate_params();
        *num_partitions = interpreter_mn2->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, /*allow_quant_ops*/ false, /*max_delegated_partitions*/ 2);

  // Three ops should be selected:
  // Add(3), Dequant(x), Add(4)
  // Since both partitions are of size 1, either could end up as the 'first'
  // partition with one Dequant node added for it.
  EXPECT_EQ(ops_to_replace->size, 3);

  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  // Verify that both Add ops have fp16 inputs.
  context->GetNodeAndRegistration(context, ops_to_replace->data[0], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  context->GetNodeAndRegistration(context, ops_to_replace->data[2], &node,
                                  &registration);
  EXPECT_EQ(context->tensors[node->inputs->data[0]].type,
            TfLiteType::kTfLiteFloat16);
  EXPECT_EQ(context->tensors[node->inputs->data[1]].type,
            TfLiteType::kTfLiteFloat16);
  // Verify that the op at index 1 is a Dequant outputing to a single Add.
  EXPECT_TRUE(ops_to_replace->data[1] == 0 || ops_to_replace->data[1] == 2);
  TfLiteIntArrayFree(ops_to_replace);
}

// Adds the pattern:
//
// float -> QUANTIZE -> ADD -> DEQUANTIZE -> float
// float -> QUANTIZE ----^
//
// The tensors between the QUANTIZE & DEQUANTIZE nodes are int8.
class InterpreterQuantized : public DelegatedInterpreter {
 public:
  InterpreterQuantized() : DelegatedInterpreter(4) {
    void* builtin_data = malloc(sizeof(int));
    EXPECT_EQ(interpreter_.AddTensors(6), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetInputs({0, 3}), kTfLiteOk);
    EXPECT_EQ(interpreter_.SetOutputs({5}), kTfLiteOk);

    // QUANTIZE 1
    const TfLiteRegistration reg_quant0 = {/*init=*/nullptr,
                                           /*free=*/nullptr,
                                           /*prepare=*/nullptr,
                                           /*invoke=*/nullptr,
                                           /*profiling_string=*/nullptr,
                                           kTfLiteBuiltinQuantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_quant0),
              kTfLiteOk);

    // QUANTIZE 2
    const TfLiteRegistration reg_quant1 = {/*init=*/nullptr,
                                           /*free=*/nullptr,
                                           /*prepare=*/nullptr,
                                           /*invoke=*/nullptr,
                                           /*profiling_string=*/nullptr,
                                           kTfLiteBuiltinQuantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{3}, /*outputs=*/{2}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_quant1),
              kTfLiteOk);

    // ADD
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
                  /*inputs=*/{1, 2}, /*outputs=*/{4}, /*init_data=*/nullptr,
                  /*init_data_size=*/0,
                  /*builtin_data=*/builtin_data,
                  /*registration=*/&reg_add0),
              kTfLiteOk);

    // DEQUANTIZE
    const TfLiteRegistration reg_dequant0 = {/*init=*/nullptr,
                                             /*free=*/nullptr,
                                             /*prepare=*/nullptr,
                                             /*invoke=*/nullptr,
                                             /*profiling_string=*/nullptr,
                                             kTfLiteBuiltinDequantize};
    EXPECT_EQ(interpreter_.AddNodeWithParameters(
                  /*inputs=*/{4}, /*outputs=*/{5}, /*init_data=*/nullptr,
                  /*init_data_size=*/0, /*builtin_data=*/nullptr,
                  /*registration=*/&reg_dequant0),
              kTfLiteOk);

    const std::vector<int> dims = {1, 3, 3, 2};

    // Input & output tensors are floating-point.
    TfLiteQuantization no_quantization;
    no_quantization.type = kTfLiteNoQuantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            0, TfLiteType::kTfLiteFloat32, "t0", dims, no_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            3, TfLiteType::kTfLiteFloat32, "t3", dims, no_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            5, TfLiteType::kTfLiteFloat32, "t5", dims, no_quantization, false),
        kTfLiteOk);
    // Other tensors are int8.
    float scale = 0.5f;
    int32_t zero_point = 12;
    TfLiteQuantization rw_quantization;
    rw_quantization.type = kTfLiteAffineQuantization;
    auto* rw_affine_quantization = static_cast<TfLiteAffineQuantization*>(
        malloc(sizeof(TfLiteAffineQuantization)));
    rw_affine_quantization->scale = TfLiteFloatArrayCreate(1);
    rw_affine_quantization->zero_point = TfLiteIntArrayCreate(1);
    rw_affine_quantization->scale->data[0] = scale;
    rw_affine_quantization->zero_point->data[0] = zero_point;
    rw_quantization.params = rw_affine_quantization;
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            1, TfLiteType::kTfLiteInt8, "t1", dims, rw_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            2, TfLiteType::kTfLiteInt8, "t2", dims, rw_quantization, false),
        kTfLiteOk);
    EXPECT_EQ(
        interpreter_.SetTensorParametersReadWrite(
            4, TfLiteType::kTfLiteInt8, "t4", dims, rw_quantization, false),
        kTfLiteOk);

    exec_plan()->data[0] = 0;
    exec_plan()->data[1] = 1;
    exec_plan()->data[2] = 2;
    exec_plan()->data[3] = 3;
  }
};

InterpreterQuantized* interpreter_quant = new InterpreterQuantized();
TEST(ModelBuilderTest, GetOpsToReplace_AllowQuantOps) {
  TfLiteContext* context = interpreter_quant->context();

  // These functions are meant to be called inside delegates. Swap out
  // for similar functions to permit direct calling of GetOpsToReplace.
  context->GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray** execution_plan) {
    *execution_plan = interpreter_quant->exec_plan();
    return kTfLiteOk;
  };
  context->GetNodeAndRegistration = [](struct TfLiteContext*, int node_index,
                                       TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    *node = interpreter_quant->node(node_index);
    *registration = interpreter_quant->registration(node_index);
    return kTfLiteOk;
  };
  context->PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array, int* num_partitions) {
        if (nodes_to_replace->size == 0) {
          *num_partitions = 0;
          return kTfLiteOk;
        }
        auto params = interpreter_quant->add_delegate_params();
        params->nodes_to_replace = TfLiteIntArrayCreate(3);
        params->nodes_to_replace->data[0] = 0;
        params->nodes_to_replace->data[1] = 1;
        params->nodes_to_replace->data[2] = 2;
        params->input_tensors = TfLiteIntArrayCreate(2);
        params->input_tensors->data[0] = 0;
        params->input_tensors->data[1] = 3;
        params->output_tensors = TfLiteIntArrayCreate(1);
        params->output_tensors->data[0] = 4;

        *partition_params_array = interpreter_quant->delegate_params();
        *num_partitions = interpreter_quant->num_delegate_params();
        return kTfLiteOk;
      };

  TfLiteIntArray* ops_to_replace =
      GetOpsToReplace(context, /**allow_quant_ops=*/true);
  // If we allow quant ops, two QUANTIZE & one ADD node should be accepted.
  EXPECT_EQ(ops_to_replace->size, 3);
  EXPECT_EQ(0, ops_to_replace->data[0]);
  EXPECT_EQ(1, ops_to_replace->data[1]);
  EXPECT_EQ(2, ops_to_replace->data[2]);

  TfLiteIntArray* ops_to_replace_without_quant =
      GetOpsToReplace(context, /**allow_quant_ops=*/false);
  // No ops should be accepted.
  EXPECT_EQ(ops_to_replace_without_quant->size, 0);

  TfLiteIntArrayFree(ops_to_replace);
  TfLiteIntArrayFree(ops_to_replace_without_quant);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
