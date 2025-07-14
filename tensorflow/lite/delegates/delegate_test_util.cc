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

#include "tensorflow/lite/delegates/delegate_test_util.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace test_utils {

TfLiteRegistration AddOpRegistration() {
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  reg.custom_name = "my_add";
  reg.builtin_code = tflite::BuiltinOperator_CUSTOM;

  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input1;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input1));
    const TfLiteTensor* input2;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &input2));
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

    // Verify that the two inputs have the same shape.
    TF_LITE_ENSURE_EQ(context, input1->dims->size, input2->dims->size);
    for (int i = 0; i < input1->dims->size; ++i) {
      TF_LITE_ENSURE_EQ(context, input1->dims->data[i], input2->dims->data[i]);
    }

    // Set output shape to match input shape.
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(
        context, output, TfLiteIntArrayCopy(input1->dims)));
    return kTfLiteOk;
  };

  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* a0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &a0));
    TF_LITE_ENSURE(context, a0);
    TF_LITE_ENSURE(context, a0->data.f);
    const TfLiteTensor* a1;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &a1));
    TF_LITE_ENSURE(context, a1);
    TF_LITE_ENSURE(context, a1->data.f);
    TfLiteTensor* out;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out));
    TF_LITE_ENSURE(context, out);
    TF_LITE_ENSURE(context, out->data.f);
    // Set output data to element-wise sum of input data.
    int num = a0->dims->data[0];
    for (int i = 0; i < num; i++) {
      out->data.f[i] = a0->data.f[i] + a1->data.f[i];
    }
    return kTfLiteOk;
  };
  return reg;
}

void TestDelegation::SetUpSubgraph(Subgraph* subgraph) {
  subgraph->AddTensors(5);
  subgraph->SetInputs({0, 1});
  subgraph->SetOutputs({3, 4});
  std::vector<int> dims({3});
  TfLiteQuantization quant{kTfLiteNoQuantization, nullptr};
  subgraph->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  TfLiteRegistration reg = AddOpRegistration();
  int node_index_ignored;
  subgraph->AddNodeWithParameters({0, 0}, {2}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({1, 1}, {3}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({2, 1}, {4}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
}

void TestDelegation::AddSubgraphs(int subgraphs_to_add,
                                  int* first_new_subgraph_index) {
  interpreter_->AddSubgraphs(subgraphs_to_add, first_new_subgraph_index);
}

void TestDelegate::SetUp() {
  interpreter_ = TestDelegation::NewInterpreterWithDefaultDelegates();
  SetUpSubgraph(&interpreter_->primary_subgraph());
}

void TestDelegate::TearDown() {
  // Interpreter relies on delegate to free the resources properly. Thus
  // the life cycle of delegate must be longer than interpreter.
  interpreter_.reset();
  delegate_.reset();
  delegate2_.reset();
}

void TestTwoDelegates::SetUp() {
  interpreter_ = TestDelegation::NewInterpreterWithDefaultDelegates();
  SetUpSubgraph(&interpreter_->primary_subgraph());
}

void TestTwoDelegates::TearDown() {
  // Interpreter relies on delegate to free the resources properly. Thus
  // the life cycle of delegate must be longer than interpreter.
  interpreter_.reset();
  delegate_.reset();
  delegate2_.reset();
}

SimpleDelegate::SimpleDelegate(const std::vector<int>& nodes,
                               int64_t delegate_flags, Options options,
                               int min_ops_per_subset)
    : nodes_(nodes),
      fail_delegate_node_init_(options & Options::kFailOnInit),
      fail_delegate_node_prepare_(options & Options::kFailOnPrepare),
      fail_delegate_node_invoke_(options & Options::kFailOnInvoke),
      min_ops_per_subset_(min_ops_per_subset),
      automatic_shape_propagation_(options ==
                                   Options::kAutomaticShapePropagation),
      custom_op_(!(options & Options::kNoCustomOp)),
      set_output_tensor_dynamic_(options & Options::kSetOutputTensorDynamic) {
  delegate_ = TfLiteDelegateCreate();
  delegate_.Prepare = [](TfLiteContext* context,
                         TfLiteDelegate* delegate) -> TfLiteStatus {
    auto* simple = static_cast<SimpleDelegate*>(delegate->data_);
    TfLiteIntArray* nodes_to_separate =
        TfLiteIntArrayCreate(simple->nodes_.size());
    // Mark nodes that we want in TfLiteIntArray* structure.
    int index = 0;
    for (auto node_index : simple->nodes_) {
      nodes_to_separate->data[index++] = node_index;
      // make sure node is added
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      if (simple->custom_op_) {
        TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
        TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
      } else {
        TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_ADD);
      }
    }
    // Check that all nodes are available
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
    for (int exec_index = 0; exec_index < execution_plan->size; exec_index++) {
      int node_index = execution_plan->data[exec_index];
      TfLiteNode* node;
      TfLiteRegistration* reg;
      context->GetNodeAndRegistration(context, node_index, &node, &reg);
      if (exec_index == node_index) {
        // Check op details only if it wasn't delegated already.
        if (simple->custom_op_) {
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
          TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
        } else {
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_ADD);
        }
      }
    }

    // Get preview of delegate partitioning from the context.
    TfLiteDelegateParams* params_array;
    int num_partitions;
    TFLITE_CHECK_EQ(
        context->PreviewDelegatePartitioning(context, nodes_to_separate,
                                             &params_array, &num_partitions),
        kTfLiteOk);

    if (simple->min_ops_per_subset() > 0) {
      // Build a new vector of ops from subsets with at least the minimum
      // size.
      std::vector<int> allowed_ops;
      for (int idx = 0; idx < num_partitions; ++idx) {
        const auto* nodes_in_subset = params_array[idx].nodes_to_replace;
        if (nodes_in_subset->size < simple->min_ops_per_subset()) continue;
        allowed_ops.insert(allowed_ops.end(), nodes_in_subset->data,
                           nodes_in_subset->data + nodes_in_subset->size);
      }

      // Free existing nodes_to_separate & initialize a new array with
      // allowed_ops.
      TfLiteIntArrayFree(nodes_to_separate);
      nodes_to_separate = TfLiteIntArrayCreate(allowed_ops.size());
      memcpy(nodes_to_separate->data, allowed_ops.data(),
             sizeof(int) * nodes_to_separate->size);
    }

    // Another call to PreviewDelegatePartitioning should be okay, since
    // partitioning memory is managed by context.
    TFLITE_CHECK_EQ(
        context->PreviewDelegatePartitioning(context, nodes_to_separate,
                                             &params_array, &num_partitions),
        kTfLiteOk);

    TfLiteStatus res = context->ReplaceNodeSubsetsWithDelegateKernels(
        context, simple->FakeFusedRegistration(), nodes_to_separate, delegate);
    TfLiteIntArrayFree(nodes_to_separate);
    return res;
  };
  delegate_.CopyToBufferHandle = [](TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteTensor* tensor) -> TfLiteStatus {
    // TODO(b/156586986): Implement tests to test buffer copying logic.
    return kTfLiteOk;
  };
  delegate_.CopyFromBufferHandle = [](TfLiteContext* context,
                                      TfLiteDelegate* delegate,
                                      TfLiteBufferHandle buffer_handle,
                                      TfLiteTensor* output) -> TfLiteStatus {
    TFLITE_CHECK_GE(buffer_handle, -1);
    TFLITE_CHECK_EQ(output->buffer_handle, buffer_handle);
    const float floats[] = {6., 6., 6.};
    int num = output->dims->data[0];
    for (int i = 0; i < num; i++) {
      output->data.f[i] = floats[i];
    }
    return kTfLiteOk;
  };

  delegate_.FreeBufferHandle =
      [](TfLiteContext* context, TfLiteDelegate* delegate,
         TfLiteBufferHandle* handle) { *handle = kTfLiteNullBufferHandle; };
  // Store type-punned data SimpleDelegate structure.
  delegate_.data_ = static_cast<void*>(this);
  delegate_.flags = delegate_flags;
}

TfLiteRegistration SimpleDelegate::FakeFusedRegistration() {
  TfLiteRegistration reg = {nullptr};
  reg.custom_name = "fake_fused_op";

  if (fail_delegate_node_init_) {
    reg.init = [](TfLiteContext* context, const char* buffer,
                  size_t length) -> void* { return TfLiteKernelInitFailed(); };
  }

  // Different flavors of the delegate kernel's Invoke(), dependent on
  // testing parameters.
  if (fail_delegate_node_invoke_) {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      return kTfLiteError;
    };
  } else {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      // Compute output data as elementwise sum of the two input arguments:
      //   func(x, y) = x + y
      // or for a single argument compute 2 * x:
      //   func(x) = x + x
      const TfLiteTensor* a0;
      const TfLiteTensor* a1;
      if (node->inputs->size == 2) {
        a0 = GetInput(context, node, 0);
        a1 = GetInput(context, node, 1);
      } else {
        a0 = GetInput(context, node, 0);
        a1 = a0;
      }
      TfLiteTensor* out;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out));
      int num = 1;
      for (int i = 0; i < a0->dims->size; ++i) {
        num *= a0->dims->data[i];
      }
      for (int i = 0; i < num; i++) {
        out->data.f[i] = a0->data.f[i] + a1->data.f[i];
      }
      if (out->buffer_handle != kTfLiteNullBufferHandle) {
        // Make the data stale so that CopyFromBufferHandle can be invoked
        out->data_is_stale = true;
      }
      return kTfLiteOk;
    };
  }

  // Different flavors of the delegate kernel's Prepare(), dependent on
  // testing parameters.
  if (automatic_shape_propagation_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Shapes should already by propagated by the runtime, just need to
      // check.
      const TfLiteTensor* input1;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input1));
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
      const int input_dims_size = input1->dims->size;
      TF_LITE_ENSURE(context, output->dims->size == input_dims_size);
      for (int i = 0; i < input_dims_size; ++i) {
        TF_LITE_ENSURE(context, output->dims->data[i] == input1->dims->data[i]);
      }
      return kTfLiteOk;
    };
  } else if (fail_delegate_node_prepare_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      return kTfLiteError;
    };
  } else if (set_output_tensor_dynamic_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
      SetTensorToDynamic(output);
      return kTfLiteOk;
    };
  } else {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Set output size to input size
      const TfLiteTensor* input1;
      const TfLiteTensor* input2;
      if (node->inputs->size == 2) {
        input1 = GetInput(context, node, 0);
        input2 = GetInput(context, node, 1);
      } else {
        input1 = GetInput(context, node, 0);
        input2 = input1;
      }
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

      TF_LITE_ENSURE_STATUS(context->ResizeTensor(
          context, output, TfLiteIntArrayCopy(input1->dims)));
      return kTfLiteOk;
    };
  }

  return reg;
}

std::unique_ptr<SimpleDelegate>
SimpleDelegate::DelegateWithRuntimeShapePropagation(
    const std::vector<int>& nodes, int64_t delegate_flags,
    int min_ops_per_subset) {
  return std::make_unique<SimpleDelegate>(
      nodes, delegate_flags,
      SimpleDelegate::Options::kAutomaticShapePropagation,
      /*min_ops_per_subset=*/min_ops_per_subset);
}

std::unique_ptr<SimpleDelegate> SimpleDelegate::DelegateWithDynamicOutput(
    const std::vector<int>& nodes) {
  // All params default except nodes & set_output_tensor_dynamic.
  return std::make_unique<SimpleDelegate>(
      nodes, kTfLiteDelegateFlagsAllowDynamicTensors,
      SimpleDelegate::Options::kSetOutputTensorDynamic);
}

void TestFP16Delegation::SetUp() {
  interpreter_ = TestDelegation::NewInterpreterWithDefaultDelegates();
  interpreter_->AddTensors(13);
  interpreter_->SetInputs({0});
  interpreter_->SetOutputs({12});

  float16_const_ = Eigen::half(2.f);

  // TENSORS.
  TfLiteQuantizationParams quant;
  // Input.
  interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Add0 output.
  interpreter_->SetTensorParametersReadOnly(
      1, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {1}, quant);
  interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Add1 output.
  interpreter_->SetTensorParametersReadOnly(
      4, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(5, kTfLiteFloat32, "", {1}, quant);
  interpreter_->SetTensorParametersReadWrite(6, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Mul0 output.
  interpreter_->SetTensorParametersReadOnly(
      7, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(8, kTfLiteFloat32, "", {1}, quant);
  interpreter_->SetTensorParametersReadWrite(9, kTfLiteFloat32, "", {1}, quant);
  // fp16 constant, dequantize output, Add2 output.
  interpreter_->SetTensorParametersReadOnly(
      10, kTfLiteFloat16, "", {1}, quant,
      reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
  interpreter_->SetTensorParametersReadWrite(11, kTfLiteFloat32, "", {1},
                                             quant);
  interpreter_->SetTensorParametersReadWrite(12, kTfLiteFloat32, "", {1},
                                             quant);

  // NODES.
  auto* add_reg = ops::builtin::Register_ADD();
  auto* mul_reg = ops::builtin::Register_MUL();
  auto* deq_reg = ops::builtin::Register_DEQUANTIZE();
  add_reg->builtin_code = kTfLiteBuiltinAdd;
  deq_reg->builtin_code = kTfLiteBuiltinDequantize;
  mul_reg->builtin_code = kTfLiteBuiltinMul;
  TfLiteAddParams* builtin_data0 =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  TfLiteAddParams* builtin_data1 =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  TfLiteMulParams* builtin_data2 =
      reinterpret_cast<TfLiteMulParams*>(malloc(sizeof(TfLiteMulParams)));
  TfLiteAddParams* builtin_data3 =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data0->activation = kTfLiteActNone;
  builtin_data1->activation = kTfLiteActNone;
  builtin_data2->activation = kTfLiteActNone;
  builtin_data3->activation = kTfLiteActNone;
  interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({0, 2}, {3}, nullptr, 0, builtin_data0,
                                      add_reg);
  interpreter_->AddNodeWithParameters({4}, {5}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({3, 5}, {6}, nullptr, 0, builtin_data1,
                                      add_reg);
  interpreter_->AddNodeWithParameters({7}, {8}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({6, 8}, {9}, nullptr, 0, builtin_data2,
                                      mul_reg);
  interpreter_->AddNodeWithParameters({10}, {11}, nullptr, 0, nullptr, deq_reg);
  interpreter_->AddNodeWithParameters({9, 11}, {12}, nullptr, 0, builtin_data3,
                                      add_reg);
}

void TestFP16Delegation::VerifyInvoke() {
  std::vector<float> input = {3.0f};
  std::vector<float> expected_output = {16.0f};

  const int input_tensor_idx = interpreter_->inputs()[0];
  const int output_tensor_idx = interpreter_->outputs()[0];

  memcpy(interpreter_->typed_tensor<float>(input_tensor_idx), input.data(),
         sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx);
  for (int i = 0; i < 1; ++i) {
    EXPECT_EQ(output_tensor->data.f[i], expected_output[i]) << i;
  }
}

TestFP16Delegation::FP16Delegate::FP16Delegate(int num_delegated_subsets,
                                               bool fail_node_prepare,
                                               bool fail_node_invoke)
    : num_delegated_subsets_(num_delegated_subsets),
      fail_delegate_node_prepare_(fail_node_prepare),
      fail_delegate_node_invoke_(fail_node_invoke) {
  delegate_.Prepare = [](TfLiteContext* context,
                         TfLiteDelegate* delegate) -> TfLiteStatus {
    auto* fp16_delegate = static_cast<FP16Delegate*>(delegate->data_);
    // FP16 graph partitioning.
    delegates::IsNodeSupportedFn node_supported_fn =
        [=](TfLiteContext* context, TfLiteNode* node,
            TfLiteRegistration* registration,
            std::string* unsupported_details) -> bool {
      return registration->builtin_code == kTfLiteBuiltinAdd;
    };
    delegates::FP16GraphPartitionHelper partition_helper(context,
                                                         node_supported_fn);
    TfLiteIntArray* nodes_to_separate = nullptr;
    if (partition_helper.Partition(nullptr) != kTfLiteOk) {
      nodes_to_separate = TfLiteIntArrayCreate(0);
    } else {
      std::vector<int> ops_to_replace =
          partition_helper.GetNodesOfFirstNLargestPartitions(
              fp16_delegate->num_delegated_subsets());
      nodes_to_separate = ConvertVectorToTfLiteIntArray(ops_to_replace);
    }

    context->ReplaceNodeSubsetsWithDelegateKernels(
        context, fp16_delegate->FakeFusedRegistration(), nodes_to_separate,
        delegate);
    TfLiteIntArrayFree(nodes_to_separate);
    return kTfLiteOk;
  };
  delegate_.CopyFromBufferHandle =
      [](TfLiteContext* context, TfLiteDelegate* delegate,
         TfLiteBufferHandle buffer_handle,
         TfLiteTensor* output) -> TfLiteStatus { return kTfLiteOk; };
  delegate_.FreeBufferHandle = nullptr;
  delegate_.CopyToBufferHandle = nullptr;
  // Store type-punned data SimpleDelegate structure.
  delegate_.data_ = static_cast<void*>(this);
  delegate_.flags = kTfLiteDelegateFlagsNone;
}

TfLiteRegistration TestFP16Delegation::FP16Delegate::FakeFusedRegistration() {
  TfLiteRegistration reg = {nullptr};
  reg.custom_name = "fake_fp16_add_op";

  // Different flavors of the delegate kernel's Invoke(), dependent on
  // testing parameters.
  if (fail_delegate_node_invoke_) {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      return kTfLiteError;
    };
  } else {
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
      float output = 0;
      for (int i = 0; i < node->inputs->size; ++i) {
        const TfLiteTensor* input_tensor = GetInput(context, node, i);
        if (input_tensor->type == kTfLiteFloat32) {
          output += input_tensor->data.f[0];
        } else {
          // All constants are 2.
          output += 2;
        }
      }
      TfLiteTensor* out = GetOutput(context, node, 0);
      out->data.f[0] = output;
      return kTfLiteOk;
    };
  }

  // Different flavors of the delegate kernel's Prepare(), dependent on
  // testing parameters.
  if (fail_delegate_node_prepare_) {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      return kTfLiteError;
    };
  } else {
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Set output size to input size
      const TfLiteTensor* input = GetInput(context, node, 0);
      TfLiteTensor* output = GetOutput(context, node, 0);
      TF_LITE_ENSURE_STATUS(context->ResizeTensor(
          context, output, TfLiteIntArrayCopy(input->dims)));
      return kTfLiteOk;
    };
  }

  return reg;
}

void TestDelegateWithControlEdges::SetUpSubgraph(Subgraph* subgraph) {
  subgraph->AddTensors(5);
  subgraph->SetInputs({0});
  subgraph->SetOutputs({4});
  std::vector<int> dims({3});
  const TfLiteQuantization quant{kTfLiteNoQuantization, nullptr};
  subgraph->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);
  subgraph->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", dims.size(),
                                         dims.data(), quant, false);

  TfLiteRegistration reg = AddOpRegistration();
  int node_index_ignored;
  subgraph->AddNodeWithParameters({0, 0}, {1}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({1, 1}, {2}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({1, 1}, {3}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
  subgraph->AddNodeWithParameters({2, 3}, {4}, {}, nullptr, 0, nullptr, &reg,
                                  &node_index_ignored);
}

}  // namespace test_utils
}  // namespace delegates
}  // namespace tflite
