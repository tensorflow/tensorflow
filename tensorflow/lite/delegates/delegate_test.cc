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

#include <stdint.h>

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/interpreter_utils.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace {

// Build a kernel registration for an op that copies its one input
// to an output
TfLiteRegistration AddOpRegistration() {
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  reg.custom_name = "my_add";
  reg.builtin_code = tflite::BuiltinOperator_CUSTOM;

  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    // Set output size to input size
    const TfLiteTensor* input1;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input1));
    const TfLiteTensor* input2;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &input2));
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

    TF_LITE_ENSURE_EQ(context, input1->dims->size, input2->dims->size);
    for (int i = 0; i < input1->dims->size; ++i) {
      TF_LITE_ENSURE_EQ(context, input1->dims->data[i], input2->dims->data[i]);
    }

    TF_LITE_ENSURE_STATUS(context->ResizeTensor(
        context, output, TfLiteIntArrayCopy(input1->dims)));
    return kTfLiteOk;
  };

  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    // Copy input data to output data.
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
    int num = a0->dims->data[0];
    for (int i = 0; i < num; i++) {
      out->data.f[i] = a0->data.f[i] + a1->data.f[i];
    }
    return kTfLiteOk;
  };
  return reg;
}

}  // namespace

// TestDelegate is a friend of Interpreter to access RemoveAllDelegates().
class TestDelegate : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);
    interpreter_->AddTensors(5);
    interpreter_->SetInputs({0, 1});
    interpreter_->SetOutputs({3, 4});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration reg = AddOpRegistration();
    interpreter_->AddNodeWithParameters({0, 0}, {2}, nullptr, 0, nullptr, &reg);
    interpreter_->AddNodeWithParameters({1, 1}, {3}, nullptr, 0, nullptr, &reg);
    interpreter_->AddNodeWithParameters({2, 1}, {4}, nullptr, 0, nullptr, &reg);
  }

  void TearDown() override {
    // Interpreter relies on delegate to free the resources properly. Thus
    // the life cycle of delegate must be longer than interpreter.
    interpreter_.reset();
    delegate_.reset();
  }

  TfLiteBufferHandle last_allocated_handle_ = kTfLiteNullBufferHandle;

  TfLiteBufferHandle AllocateBufferHandle() { return ++last_allocated_handle_; }

  TfLiteStatus RemoveAllDelegates() {
    return interpreter_->RemoveAllDelegates();
  }

 protected:
  class SimpleDelegate {
   public:
    // Create a simple implementation of a TfLiteDelegate. We use the C++ class
    // SimpleDelegate and it can produce a handle TfLiteDelegate that is
    // value-copyable and compatible with TfLite.
    // fail_node_prepare: To simulate failure of Delegate node's Prepare().
    // min_ops_per_subset: If >0, partitioning preview is used to choose only
    // those subsets with min_ops_per_subset number of nodes.
    // fail_node_invoke: To simulate failure of Delegate node's Invoke().
    // automatic_shape_propagation: This assumes that the runtime will propagate
    // shapes using the original execution plan.
    explicit SimpleDelegate(const std::vector<int>& nodes,
                            int64_t delegate_flags = kTfLiteDelegateFlagsNone,
                            bool fail_node_prepare = false,
                            int min_ops_per_subset = 0,
                            bool fail_node_invoke = false,
                            bool automatic_shape_propagation = false)
        : nodes_(nodes),
          fail_delegate_node_prepare_(fail_node_prepare),
          min_ops_per_subset_(min_ops_per_subset),
          fail_delegate_node_invoke_(fail_node_invoke),
          automatic_shape_propagation_(automatic_shape_propagation) {
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
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
          TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
        }
        // Check that all nodes are available
        TfLiteIntArray* execution_plan;
        TF_LITE_ENSURE_STATUS(
            context->GetExecutionPlan(context, &execution_plan));
        for (int exec_index = 0; exec_index < execution_plan->size;
             exec_index++) {
          int node_index = execution_plan->data[exec_index];
          TfLiteNode* node;
          TfLiteRegistration* reg;
          context->GetNodeAndRegistration(context, node_index, &node, &reg);
          if (exec_index == node_index) {
            // Check op details only if it wasn't delegated already.
            TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
            TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
          }
        }

        // Get preview of delegate partitioning from the context.
        TfLiteDelegateParams* params_array;
        int num_partitions;
        TFLITE_CHECK_EQ(
            context->PreviewDelegatePartitioning(
                context, nodes_to_separate, &params_array, &num_partitions),
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

        // Another call to PreviewDelegateParitioning should be okay, since
        // partitioning memory is managed by context.
        TFLITE_CHECK_EQ(
            context->PreviewDelegatePartitioning(
                context, nodes_to_separate, &params_array, &num_partitions),
            kTfLiteOk);

        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, simple->FakeFusedRegistration(), nodes_to_separate,
            delegate);
        TfLiteIntArrayFree(nodes_to_separate);
        return kTfLiteOk;
      };
      delegate_.CopyToBufferHandle = [](TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) -> TfLiteStatus {
        // TODO(b/156586986): Implement tests to test buffer copying logic.
        return kTfLiteOk;
      };
      delegate_.CopyFromBufferHandle =
          [](TfLiteContext* context, TfLiteDelegate* delegate,
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

    TfLiteRegistration FakeFusedRegistration() {
      TfLiteRegistration reg = {nullptr};
      reg.custom_name = "fake_fused_op";

      // Different flavors of the delegate kernel's Invoke(), dependent on
      // testing parameters.
      if (fail_delegate_node_invoke_) {
        reg.invoke = [](TfLiteContext* context,
                        TfLiteNode* node) -> TfLiteStatus {
          return kTfLiteError;
        };
      } else {
        reg.invoke = [](TfLiteContext* context,
                        TfLiteNode* node) -> TfLiteStatus {
          // Copy input data to output data.
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
            TF_LITE_ENSURE(context,
                           output->dims->data[i] == input1->dims->data[i]);
          }
          return kTfLiteOk;
        };
      } else if (fail_delegate_node_prepare_) {
        reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
          return kTfLiteError;
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

    TfLiteDelegate* get_tf_lite_delegate() { return &delegate_; }

    int min_ops_per_subset() { return min_ops_per_subset_; }

   private:
    std::vector<int> nodes_;
    TfLiteDelegate delegate_;
    bool fail_delegate_node_prepare_ = false;
    int min_ops_per_subset_ = 0;
    bool fail_delegate_node_invoke_ = false;
    bool automatic_shape_propagation_ = false;
  };

  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<SimpleDelegate> delegate_, delegate2_;
};
namespace {

TEST_F(TestDelegate, BasicDelegate) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  int node = interpreter_->execution_plan()[0];
  const auto* node_and_reg = interpreter_->node_and_registration(node);
  EXPECT_EQ(node_and_reg->second.custom_name,
            delegate_->FakeFusedRegistration().custom_name);

  const TfLiteDelegateParams* params = static_cast<const TfLiteDelegateParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_EQ(params->nodes_to_replace->size, 3);
  EXPECT_EQ(params->nodes_to_replace->data[0], 0);
  EXPECT_EQ(params->nodes_to_replace->data[1], 1);
  EXPECT_EQ(params->nodes_to_replace->data[2], 2);

  ASSERT_EQ(params->input_tensors->size, 2);
  EXPECT_EQ(params->input_tensors->data[0], 0);
  EXPECT_EQ(params->input_tensors->data[1], 1);

  ASSERT_EQ(params->output_tensors->size, 2);
  EXPECT_EQ(params->output_tensors->data[0], 3);
  EXPECT_EQ(params->output_tensors->data[1], 4);
}

TEST_F(TestDelegate, DelegateNodePrepareFailure) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, kTfLiteDelegateFlagsNone, true /**fail_node_prepare**/));
  // ModifyGraphWithDelegate fails, since the Prepare() method in the node's
  // TfLiteRegistration returns an error status.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
  // Execution plan should remain unchanged.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, DelegateNodeInvokeFailure) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Delegation modified execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;

  // Verify Invoke() behavior: fails first, succeeds after RemoveAllDelegates().
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  ASSERT_EQ(RemoveAllDelegates(), kTfLiteOk);
  // Delegation removed, returning to original execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, DelegateNodeInvokeFailureFallback) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Delegation modified execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(
      delegates::InterpreterUtils::InvokeWithCPUFallback(interpreter_.get()),
      kTfLiteDelegateError);
  // Delegation removed, returning to original execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  // Check outputs.
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, SecondDelegationPrepareFailure) {
  // First delegate only supports nodes 1, 2. Gets applied successfully.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({1, 2}, kTfLiteDelegateFlagsAllowDynamicTensors));
  // Second delegate supports node 0, but fails during the delegate-node's
  // Prepare.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0}, kTfLiteDelegateFlagsNone, true /**fail_node_prepare**/));

  // Initially, execution plan has 3 nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  // First delegate should be applied successfully, yielding a plan with 2
  // nodes.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  // Second delegate won't get applied.
  // As a result, previous delegate should also get undone, restoring the
  // execution plan to its original state.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, SecondDelegationInvokeFailure) {
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({1, 2}, kTfLiteDelegateFlagsAllowDynamicTensors));
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  // Outputs match the AddOp path, rather than delegate path.
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;

  // Verify Invoke() behavior to ensure Interpreter isn't broken.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  EXPECT_EQ(RemoveAllDelegates(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// This test ensures that node indices in multi-delegate application are handled
// correctly by the TFLite partitioning algorithm.
TEST_F(TestDelegate, TwoDelegates_ExecutionPlanIndicesDifferent) {
  // First delegate supports nodes 0, 1.
  // After this delegation, the execution plan size is 2.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({0, 1}, kTfLiteDelegateFlagsAllowDynamicTensors));
  // Second delegate supports (original) node index 2.
  // The execution plan has 2 nodes, so this verifies that the partitioning
  // algorithm correctly refers to (original) node indices instead of execution
  // plan indices.
  delegate2_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({2}, kTfLiteDelegateFlagsNone));

  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Verify Invoke works.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
}

TEST_F(TestDelegate, StaticDelegateMakesGraphImmutable) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  // Deliberately try to set tensor params with quantization while immutable,
  // ensuring quantization is properly freed.
  TfLiteQuantization quant = {};
  quant.type = kTfLiteAffineQuantization;
  auto quant_params = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = nullptr;
  quant_params->zero_point = nullptr;
  quant_params->quantized_dimension = 0;
  quant.params = quant_params;
  ASSERT_NE(interpreter_->SetTensorParametersReadWrite(0, kTfLiteInt8, "", {3},
                                                       quant),
            kTfLiteOk);
}

TEST_F(TestDelegate, ComplexDelegate) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({1, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  // 0th should be a non-delegated original op
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
  // 1st should be a new macro op (3) which didn't exist)
  ASSERT_EQ(interpreter_->execution_plan()[1], 3);
  const auto* node_and_reg = interpreter_->node_and_registration(3);
  ASSERT_EQ(node_and_reg->second.custom_name,
            delegate_->FakeFusedRegistration().custom_name);
}

TEST_F(TestDelegate, SetBufferHandleToInput) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 0;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  ASSERT_EQ(tensor->delegate, nullptr);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
}

TEST_F(TestDelegate, SetBufferHandleToOutput) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
}

TEST_F(TestDelegate, SetInvalidHandleToTensor) {
  interpreter_->Invoke();
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  SimpleDelegate another_simple_delegate({0, 1, 2});

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status = interpreter_->SetBufferHandle(
      kOutputTensorIndex, handle,
      another_simple_delegate.get_tf_lite_delegate());
  // Setting a buffer handle to a tensor with another delegate will fail.
  ASSERT_EQ(status, kTfLiteError);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);
}

// We utilize delegation in such a way as to allow node subsets with a minimum
// number of ops only.
TEST_F(TestDelegate, TestDelegationWithPartitionPreview) {
  // We set kTfLiteDelegateFlagsAllowDynamicTensors to ensure the second
  // delegate can be applied.
  // Ops 0 and 2 are delegated but end up in the same partition (based on
  // dependency analysis). However, since min_ops_per_subset = 3, no delegation
  // takes place.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 2}, kTfLiteDelegateFlagsAllowDynamicTensors,
      false /**fail_node_prepare**/, 3 /**min_ops_per_subset**/));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  // Original execution plan remains.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
  ASSERT_EQ(interpreter_->execution_plan()[1], 1);
  ASSERT_EQ(interpreter_->execution_plan()[2], 2);

  // Same ops supported, but min_ops_per_subset = 2.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 2}, kTfLiteDelegateFlagsAllowDynamicTensors,
      false /**fail_node_prepare**/, 2 /**min_ops_per_subset**/));
  interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  ASSERT_EQ(interpreter_->execution_plan()[0], 3);
  const auto* node_and_reg = interpreter_->node_and_registration(3);
  ASSERT_EQ(node_and_reg->second.custom_name,
            delegate2_->FakeFusedRegistration().custom_name);
  ASSERT_EQ(interpreter_->execution_plan()[1], 1);
}

TEST_F(TestDelegate, TestResizeInputWithNonDynamicDelegate) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  // Try resizing input to same shape as before (which should be a No-op).
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  // This should fail, since the previous application of the delegate will be
  // re-done automatically, making the graph immutable again.
  ASSERT_NE(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Ensure graph has been restored to its valid delegated state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }

  // Resize again, but call AllocateTensors as usual afterwards.
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, TestResizeInputWithMultipleDelegates) {
  // First delegate only supports node 0.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({0}, kTfLiteDelegateFlagsAllowDynamicTensors));
  // Second delegate supports nodes 1 & 2, and makes the graph immutable.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({1, 2}));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegates nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Try resizing input to same shape as before (which should be a No-op).
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Resizing input tensors should temporarily restore original execution plan
  // of 3 nodes.
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  // This should fail, since the previous application of the delegate will be
  // re-done automatically, making the graph immutable again.
  ASSERT_NE(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Ensure graph has been restored to its valid delegated state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }

  // Resize again, but call AllocateTensors as usual afterwards.
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// If a delegate sets kTfLiteDelegateFlagsRequirePropagatedShapes but not
// kTfLiteDelegateFlagsAllowDynamicTensors, the former is redundant.
TEST_F(TestDelegate, TestRequirePropagatedShapes_NonDynamicDelegate) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, kTfLiteDelegateFlagsRequirePropagatedShapes));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  // Resizing should revert execution plan to original state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, TestRequirePropagatedShapes_DynamicDelegateWithFlag) {
  // Delegate sets both flags and in its Prepare, ensures that shapes have been
  // propagated by runtime.
  int delegate_flags = kTfLiteDelegateFlagsAllowDynamicTensors |
                       kTfLiteDelegateFlagsRequirePropagatedShapes;
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, delegate_flags, false /**fail_node_prepare**/,
      3 /**min_ops_per_subset**/, false /**fail_node_invoke**/,
      true /**automatic_shape_propagation**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// If the delegate implementation expects shapes to be automatically propagated
// but does not set the required flag, its Prepare should fail.
TEST_F(TestDelegate, TestRequirePropagatedShapes_DynamicDelegateWithoutFlag) {
  // Delegate sets both flags and in its Prepare, ensures that shapes have been
  // propagated by runtime.
  int delegate_flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1, 2}, delegate_flags, false /**fail_node_prepare**/,
      3 /**min_ops_per_subset**/, false /**fail_node_invoke**/,
      true /**automatic_shape_propagation**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
}

TEST_F(TestDelegate, TestRequirePropagatedShapes_MultipleDelegates) {
  // First delegate needs to support dynamic tensors to allow second delegation.
  // This delegate does not require automatic propagation.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0}, kTfLiteDelegateFlagsAllowDynamicTensors,
      false /**fail_node_prepare**/, 1 /**min_ops_per_subset**/,
      false /**fail_node_invoke**/, false /**automatic_shape_propagation**/));
  // Second delegate supports nodes 1 & 2, and requires automatic shape
  // propagation.
  int delegate_flags = kTfLiteDelegateFlagsAllowDynamicTensors |
                       kTfLiteDelegateFlagsRequirePropagatedShapes;
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {1, 2}, delegate_flags, false /**fail_node_prepare**/,
      1 /**min_ops_per_subset**/, false /**fail_node_invoke**/,
      true /**automatic_shape_propagation**/));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegate nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, TestFallbackWithMultipleDelegates) {
  // First delegate only supports node 0.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({0}, kTfLiteDelegateFlagsAllowDynamicTensors));
  // Second delegate supports nodes 1 & 2, and makes the graph immutable.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {1, 2}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/));
  // Pre-delegation execution plan should have three nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegates nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(
      delegates::InterpreterUtils::InvokeWithCPUFallback(interpreter_.get()),
      kTfLiteDelegateError);
  // All delegates should be undone.
  EXPECT_EQ(interpreter_->execution_plan().size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, ReleaseNonPersistentMemoryWithDelegates) {
  // First delegate only supports node 0.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({0}, kTfLiteDelegateFlagsAllowDynamicTensors));
  // Second delegate supports nodes 1 & 2, and makes the graph immutable.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({1, 2}));

  // No-op.
  ASSERT_EQ(interpreter_->ReleaseNonPersistentMemory(), kTfLiteOk);

  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegates nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  ASSERT_EQ(interpreter_->ReleaseNonPersistentMemory(), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // This should fail, since the graph is immutable.
  ASSERT_NE(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  interpreter_->Invoke();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }

  ASSERT_EQ(interpreter_->ReleaseNonPersistentMemory(), kTfLiteOk);
}

TEST_F(TestDelegate, TestCopyFromBufferInvoke) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  std::vector<float> floats = {1.0f, 2.0f, 3.0f};
  memcpy(interpreter_->typed_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  memcpy(interpreter_->typed_tensor<float>(1), floats.data(),
         floats.size() * sizeof(float));

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  // Called Invoke without setting the buffer will not call the CopyFromBuffer
  interpreter_->Invoke();
  std::vector<float> res = {2.0f, 4.0f, 6.0f};
  for (int i = 0; i < tensor->dims->data[0]; ++i) {
    ASSERT_EQ(tensor->data.f[i], res[i]);
  }
}

TEST_F(TestDelegate, TestCopyFromBuffer) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  std::vector<float> floats = {1.0f, 2.0f, 3.0f};
  memcpy(interpreter_->typed_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  memcpy(interpreter_->typed_tensor<float>(1), floats.data(),
         floats.size() * sizeof(float));

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  interpreter_->Invoke();
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
  for (int i = 0; i < tensor->dims->data[0]; ++i) {
    ASSERT_EQ(tensor->data.f[i], 6.0f);
  }
}

TEST_F(TestDelegate, DelegateCustomOpResolution) {
  // Build a flatbuffer model that contains the "my_add" custom op which gets
  // resolved only after SimpleDelegate is applied.
  flatbuffers::FlatBufferBuilder builder;
  // Tensors.
  const int32_t shape[1] = {3};
  flatbuffers::Offset<Tensor> tensors[3] = {
      CreateTensor(builder, builder.CreateVector<int32_t>(shape, 1),
                   TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("X")),
      CreateTensor(builder, builder.CreateVector<int32_t>(shape, 1),
                   TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("Y")),
      CreateTensor(builder, builder.CreateVector<int32_t>(shape, 1),
                   TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("Z")),
  };
  // Custom op definition.
  flatbuffers::Offset<OperatorCode> op_code =
      CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "my_add");
  const int32_t inputs[2] = {0, 1};
  const int32_t outputs[1] = {2};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>(inputs, 2),
      builder.CreateVector<int32_t>(outputs, 1), BuiltinOptions_NONE,
      /*builtin_options=*/0,
      /*custom_options=*/0, tflite::CustomOptionsFormat_FLEXBUFFERS);
  // Subgraph & Model.
  flatbuffers::Offset<SubGraph> subgraph =
      CreateSubGraph(builder, builder.CreateVector(tensors, 3),
                     builder.CreateVector<int32_t>(inputs, 2),
                     builder.CreateVector<int32_t>(outputs, 1),
                     builder.CreateVector(&op, 1), /*name=*/0);
  flatbuffers::Offset<Buffer> buffers[1] = {
      CreateBuffer(builder, builder.CreateVector({})),
  };
  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&op_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("test_model"),
      builder.CreateVector(buffers, 1));
  builder.Finish(model_buffer);
  std::vector<char> buffer =
      std::vector<char>(builder.GetBufferPointer(),
                        builder.GetBufferPointer() + builder.GetSize());
  const Model* model = GetModel(buffer.data());

  // Build an interpreter with the model. Initialization should work fine.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model, ::tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  // AllocateTensors should fail, since my_add hasn't been resolved.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteError);

  // Applying static delegate won't work, since the interpreter will first try
  // to Prepare all original nodes.
  std::unique_ptr<SimpleDelegate> static_delegate(new SimpleDelegate({0}));
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(
                static_delegate->get_tf_lite_delegate()),
            kTfLiteError);

  // Applying delegate that supports dynamic tensors should work.
  std::unique_ptr<SimpleDelegate> dynamic_delegate(
      new SimpleDelegate({0}, kTfLiteDelegateFlagsAllowDynamicTensors));
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(
                dynamic_delegate->get_tf_lite_delegate()),
            kTfLiteOk);
  // AllocateTensors will now work.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
}

class TestDelegateWithDynamicTensors : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);

    interpreter_->AddTensors(3);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({1, 2});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration reg = DynamicCopyOpRegistration();
    interpreter_->AddNodeWithParameters({0}, {1, 2}, nullptr, 0, nullptr, &reg);

    delegate_.Prepare = [](TfLiteContext* context,
                           TfLiteDelegate* delegate) -> TfLiteStatus {
      // In this test, the delegate replaces all the nodes if this function is
      // called.
      TfLiteIntArray* execution_plan;
      TF_LITE_ENSURE_STATUS(
          context->GetExecutionPlan(context, &execution_plan));
      context->ReplaceNodeSubsetsWithDelegateKernels(
          context, DelegateRegistration(), execution_plan, delegate);
      return kTfLiteOk;
    };
    delegate_.flags = kTfLiteDelegateFlagsNone;
  }

  static TfLiteRegistration DynamicCopyOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Output 0 is dynamic
      TfLiteTensor* output0;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output0));
      SetTensorToDynamic(output0);
      // Output 1 has the same shape as input.
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
      TfLiteTensor* output1;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output1));
      TF_LITE_ENSURE_STATUS(context->ResizeTensor(
          context, output1, TfLiteIntArrayCopy(input->dims)));
      return kTfLiteOk;
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      // Not implemented since this isn't required in testing.
      return kTfLiteOk;
    };
    return reg;
  }

  static TfLiteRegistration DelegateRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // If tensors are resized, the runtime should propagate shapes
      // automatically if correct flag is set. Ensure values are correct.
      // Output 0 should be dynamic.
      TfLiteTensor* output0;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output0));
      TF_LITE_ENSURE(context, IsDynamicTensor(output0));
      // Output 1 has the same shape as input.
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
      TfLiteTensor* output1;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output1));
      TF_LITE_ENSURE(context, input->dims->size == output1->dims->size);
      TF_LITE_ENSURE(context, input->dims->data[0] == output1->dims->data[0]);
      return kTfLiteOk;
    };

    return reg;
  }

  std::unique_ptr<Interpreter> interpreter_;
  TfLiteDelegate delegate_;
};

TEST_F(TestDelegateWithDynamicTensors, DisallowDynamicTensors) {
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The interpreter should not call delegate's `Prepare` when dynamic tensors
  // exist. So the node ID isn't changed.
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
}

TEST_F(TestDelegateWithDynamicTensors, AllowDynamicTensors) {
  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The node should be replaced because dynamic tensors are allowed. Therefore
  // only node ID in the execution plan is changed from 0 to 1.
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);
}

TEST_F(TestDelegateWithDynamicTensors, ModifyGraphAfterAllocate) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);

  // Allocation should still succeed.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestDelegateWithDynamicTensors, ShapePropagation_FlagSet) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors |
                    kTfLiteDelegateFlagsRequirePropagatedShapes;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  // Allocation before & after resizing tensors should work.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestDelegateWithDynamicTensors, ShapePropagation_FlagNotSet) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  // Allocation after resizing tensors should NOT work, since runtime won't
  // propagate shape - causing delegate kernel to fail.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
}

// Tests for FP16 graphs
// =====================

// Tests delegate functionality related to FP16 graphs.
// Model architecture:
// 1->DEQ->2   4->DEQ->5   7->DEQ->8   10->DEQ->11
//         |           |           |            |
// 0----->ADD->3----->ADD->6----->MUL->9------>ADD-->12
// Input: 0, Output:12.
// All constants are 2, so the function is: (x + 2 + 2) * 2 + 2 = 2x + 10
//
// Delegate only supports ADD, so can have up to two delegated partitions.
// TODO(b/156707497): Add more cases here once we have landed CPU kernels
// supporting FP16.
class TestFP16Delegation : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);
    interpreter_->AddTensors(13);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({12});

    float16_const_ = Eigen::half_impl::float_to_half_rtne(2.f);

    // TENSORS.
    TfLiteQuantizationParams quant;
    // Input.
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {1},
                                               quant);
    // fp16 constant, dequantize output, Add0 output.
    interpreter_->SetTensorParametersReadOnly(
        1, kTfLiteFloat16, "", {1}, quant,
        reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {1},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {1},
                                               quant);
    // fp16 constant, dequantize output, Add1 output.
    interpreter_->SetTensorParametersReadOnly(
        4, kTfLiteFloat16, "", {1}, quant,
        reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
    interpreter_->SetTensorParametersReadWrite(5, kTfLiteFloat32, "", {1},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(6, kTfLiteFloat32, "", {1},
                                               quant);
    // fp16 constant, dequantize output, Mul0 output.
    interpreter_->SetTensorParametersReadOnly(
        7, kTfLiteFloat16, "", {1}, quant,
        reinterpret_cast<const char*>(&float16_const_), sizeof(TfLiteFloat16));
    interpreter_->SetTensorParametersReadWrite(8, kTfLiteFloat32, "", {1},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(9, kTfLiteFloat32, "", {1},
                                               quant);
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
    interpreter_->AddNodeWithParameters({10}, {11}, nullptr, 0, nullptr,
                                        deq_reg);
    interpreter_->AddNodeWithParameters({9, 11}, {12}, nullptr, 0,
                                        builtin_data3, add_reg);
  }

  void VerifyInvoke() {
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

  void TearDown() override { interpreter_.reset(); }

 protected:
  class FP16Delegate {
   public:
    // Uses FP16GraphPartitionHelper to accept ADD nodes with fp16 input.
    explicit FP16Delegate(int num_delegated_subsets,
                          bool fail_node_prepare = false,
                          bool fail_node_invoke = false)
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

    TfLiteRegistration FakeFusedRegistration() {
      TfLiteRegistration reg = {nullptr};
      reg.custom_name = "fake_fp16_add_op";

      // Different flavors of the delegate kernel's Invoke(), dependent on
      // testing parameters.
      if (fail_delegate_node_invoke_) {
        reg.invoke = [](TfLiteContext* context,
                        TfLiteNode* node) -> TfLiteStatus {
          return kTfLiteError;
        };
      } else {
        reg.invoke = [](TfLiteContext* context,
                        TfLiteNode* node) -> TfLiteStatus {
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

    TfLiteDelegate* get_tf_lite_delegate() { return &delegate_; }

    int num_delegated_subsets() { return num_delegated_subsets_; }

   private:
    TfLiteDelegate delegate_;
    int num_delegated_subsets_;
    bool fail_delegate_node_prepare_ = false;
    bool fail_delegate_node_invoke_ = false;
  };

  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<FP16Delegate> delegate_;
  Eigen::half float16_const_;
};

TEST_P(TestFP16Delegation, NonDelegatedInterpreterWorks) {
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

TEST_P(TestFP16Delegation, DelegationWorks) {
  delegate_ = std::unique_ptr<FP16Delegate>(
      new FP16Delegate(/**num_delegated_subsets**/ GetParam()));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should have 5 nodes: delegate, mul, add2 & 2 dequantize (one for mul &
  // add2).
  ASSERT_EQ(interpreter_->execution_plan().size(), 5);
  VerifyInvoke();
}

TEST_P(TestFP16Delegation, DelegatePrepareFails) {
  delegate_ = std::unique_ptr<FP16Delegate>(new FP16Delegate(
      /**num_delegated_subsets**/ GetParam(), /**fail_node_prepare**/ true));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
  // Delegation failed, but runtime should go back to correct previous state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 8);
  VerifyInvoke();
}

TEST_P(TestFP16Delegation, DelegateInvokeWithCPUFallback) {
  delegate_ = std::unique_ptr<FP16Delegate>(new FP16Delegate(
      /**num_delegated_subsets**/ GetParam(), /**fail_node_prepare**/ false,
      /**fail_node_invoke**/ true));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  std::vector<float> input = {3.0f};
  std::vector<float> expected_output = {16.0f};

  const int input_tensor_idx = interpreter_->inputs()[0];
  const int output_tensor_idx = interpreter_->outputs()[0];

  memcpy(interpreter_->typed_tensor<float>(input_tensor_idx), input.data(),
         sizeof(float));
  EXPECT_EQ(
      delegates::InterpreterUtils::InvokeWithCPUFallback(interpreter_.get()),
      kTfLiteDelegateError);
  TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx);
  for (int i = 0; i < 1; ++i) {
    EXPECT_EQ(output_tensor->data.f[i], expected_output[i]) << i;
  }

  ASSERT_EQ(interpreter_->execution_plan().size(), 8);
  VerifyInvoke();
}

INSTANTIATE_TEST_SUITE_P(TestFP16Delegation, TestFP16Delegation,
                         ::testing::Values(1, 2));

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
