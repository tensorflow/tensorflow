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

#ifndef TENSORFLOW_LITE_INTERPRETER_TEST_H_
#define TENSORFLOW_LITE_INTERPRETER_TEST_H_

#include <gtest/gtest.h>
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
// InterpreterTest is a friend of Interpreter, so it can access context_.
class InterpreterTest : public ::testing::Test {
 public:
  template <typename Delegate>
  static TfLiteStatus ModifyGraphWithDelegate(
      Interpreter* interpreter, std::unique_ptr<Delegate> delegate) {
    Interpreter::TfLiteDelegatePtr tflite_delegate(
        delegate.release(), [](TfLiteDelegate* delegate) {
          delete reinterpret_cast<Delegate*>(delegate);
        });
    return interpreter->ModifyGraphWithDelegate(std::move(tflite_delegate));
  }

 protected:
  TfLiteContext* GetInterpreterContext() { return interpreter_.context_; }

  Interpreter interpreter_;
};

// Build a kernel registration for an op that copies its one input
// to an output
TfLiteRegistration AddOpRegistration() {
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  reg.custom_name = "my_add";
  reg.builtin_code = tflite::BuiltinOperator_CUSTOM;

  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    // Set output size to input size
    const TfLiteTensor* input1 = GetInput(context, node, 0);
    const TfLiteTensor* input2 = GetInput(context, node, 1);
    TfLiteTensor* output = GetOutput(context, node, 0);

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
    const TfLiteTensor* a0 = GetInput(context, node, 0);
    TF_LITE_ENSURE(context, a0);
    TF_LITE_ENSURE(context, a0->data.f);
    const TfLiteTensor* a1 = GetInput(context, node, 1);
    TF_LITE_ENSURE(context, a1);
    TF_LITE_ENSURE(context, a1->data.f);
    TfLiteTensor* out = GetOutput(context, node, 0);
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
    explicit SimpleDelegate(
        const std::vector<int>& nodes,
        TfLiteDelegateFlags delegate_flags = kTfLiteDelegateFlagsNone,
        bool fail_node_prepare = false, int min_ops_per_subset = 0,
        bool fail_node_invoke = false)
        : nodes_(nodes),
          fail_delegate_node_prepare_(fail_node_prepare),
          min_ops_per_subset_(min_ops_per_subset),
          fail_delegate_node_invoke_(fail_node_invoke) {
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
          // Build a new vector of ops from subsets with atleast the minimum
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
        TfLiteTensor* out = GetOutput(context, node, 0);
        int num = 1;
        for (int i = 0; i < a0->dims->size; ++i) {
          num *= a0->dims->data[i];
        }
        for (int i = 0; i < num; i++) {
          out->data.f[i] = a0->data.f[i] + a1->data.f[i];
        }
        // Make the data stale so that CopyFromBufferHandle can be invoked
        if (out->buffer_handle != kTfLiteNullBufferHandle) {
          out->data_is_stale = true;
        }
        return kTfLiteOk;
      };
      if (fail_delegate_node_invoke_) {
        reg.invoke = [](TfLiteContext* context,
                        TfLiteNode* node) -> TfLiteStatus {
          return kTfLiteError;
        };
      }

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
        TfLiteTensor* output = GetOutput(context, node, 0);

        TF_LITE_ENSURE_STATUS(context->ResizeTensor(
            context, output, TfLiteIntArrayCopy(input1->dims)));
        return kTfLiteOk;
      };
      if (fail_delegate_node_prepare_) {
        reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
          return kTfLiteError;
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
  };

  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<SimpleDelegate> delegate_, delegate2_;
};
}  // namespace tflite

#endif  // TENSORFLOW_LITE_INTERPRETER_TEST_H_
