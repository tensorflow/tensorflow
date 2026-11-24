/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/subgraph.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/util.h"

namespace tflite {

namespace ops {
namespace builtin {
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_NEG();
}  // namespace builtin
}  // namespace ops

namespace {

using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::Not;

TEST(RemoveUnusedInputs, NothingToRemove) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(4);
  subgraph.SetInputs({0, 1});
  subgraph.SetOutputs({3});
  TfLiteRegistration* pad_op = tflite::ops::builtin::Register_PADV2();
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  subgraph.AddNodeWithParameters({0, 1}, {2}, {}, nullptr, 0, nullptr, pad_op);
  subgraph.AddNodeWithParameters({2}, {3}, {}, nullptr, 0, nullptr, neg_op);

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({0, 1}));
}

TEST(RemoveUnusedInputs, HasUnusedInputs) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(4);
  subgraph.SetInputs({0, 1, 2});
  subgraph.SetOutputs({3});
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  subgraph.AddNodeWithParameters({2}, {3}, {}, nullptr, 0, nullptr, neg_op);

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({-1, -1, 2}));
}

TEST(RemoveUnusedInputs, BypassInputsWithoutOp) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(3);
  subgraph.SetInputs({0, 1, 2});
  subgraph.SetOutputs({0, 2});

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({0, -1, 2}));
}

TEST(GetSubgraphContext, NonConstGetSubgraphContext) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  TfLiteContext* context = nullptr;

  EXPECT_EQ(kTfLiteError, subgraph.AcquireSubgraphContext(-1, &context));
  ASSERT_EQ(context, nullptr);

  EXPECT_EQ(kTfLiteError, subgraph.AcquireSubgraphContext(1, &context));
  ASSERT_EQ(context, nullptr);

  EXPECT_EQ(kTfLiteOk, subgraph.AcquireSubgraphContext(0, &context));
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(kTfLiteOk, subgraph.ReleaseSubgraphContext(0));
}

TEST(MarkSubgraphAsDelegationSkippable, MarkSubgraphAsDelegationSkippable) {
  static StderrReporter* error_reporter = new StderrReporter;
  // Construct a mock subgraph vector with two entries.
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 2; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/error_reporter,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }

  // The primary subgraph shouldn't be delegation-skippable.
  ASSERT_EQ(subgraphs[0]->MarkSubgraphAsDelegationSkippable(0), kTfLiteError);
  ASSERT_FALSE(subgraphs[0]->IsDelegationSkippable());

  // The subgraph_index shouldn't exceed the total number of subgraphs.
  ASSERT_EQ(subgraphs[0]->MarkSubgraphAsDelegationSkippable(2), kTfLiteError);

  ASSERT_EQ(subgraphs[0]->MarkSubgraphAsDelegationSkippable(1), kTfLiteOk);
  ASSERT_TRUE(subgraphs[1]->IsDelegationSkippable());
}

// Helper to get the minimal buffer size to allocate for a buffer of given
// shape.
size_t BytesFor(const TfLiteType type, const int* const data,
                const size_t size) {
  size_t type_size;
  CHECK_EQ(GetSizeOfType(nullptr, type, &type_size), kTfLiteOk)
      << "Type is not supported by GetSizeOfType";
  return std::accumulate(data, data + size, type_size, std::multiplies<int>());
}

size_t BytesFor(const TfLiteType type, const TfLiteIntArray& dims) {
  return BytesFor(type, dims.data, dims.size);
}

size_t BytesFor(const TfLiteType type, const std::vector<int>& dims) {
  return BytesFor(type, dims.data(), dims.size());
}

// Sets up a TFLite context and default values to initialize/resize test
// tensors.
class SubgraphResizeTensorTest : public testing::Test {
 public:
  SubgraphResizeTensorTest() {
    tensor_.type = type_;
    tensor_.allocation_type = kTfLiteDynamic;
  }

  ~SubgraphResizeTensorTest() override { TfLiteTensorFree(&tensor_); }

 protected:
  const TfLiteType type_ = kTfLiteInt32;
  Interpreter interpreter_;
  TfLiteContext& context_ = *interpreter_.primary_subgraph().context();
  const std::vector<int> reference_shape_ = {5, 4, 3};
  const size_t reference_dims_bytes_ = BytesFor(type_, reference_shape_);
  TfLiteTensor tensor_ = {};
  TfLiteIntArray* dims_ = ConvertVectorToTfLiteIntArray(reference_shape_);
};

TEST_F(SubgraphResizeTensorTest, ResizeEmptyDynamicTensorAllocateData) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  EXPECT_EQ(tensor_.dims, dims_);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  // Touch memory to trigger ASAN in case of failure.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeEmptyDynamicTensorWithStoredShapeAllocatesData) {
  tensor_.dims = dims_;
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest, ResizeDynamicTensorWithTheEqualShapeIsANoop) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  const void* const initial_data = tensor_.data.data;

  TfLiteIntArray* dims2 = ConvertVectorToTfLiteIntArray(reference_shape_);
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims2), kTfLiteOk);

  EXPECT_EQ(tensor_.dims, dims2);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_GE(tensor_.data.data, initial_data);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest, ResizeDynamicTensorWithStoredShapeIsANoop) {
  tensor_.dims = dims_;
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);
  const void* const initial_data = tensor_.data.data;
  // Reallocate the tensor with its current shape.
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_GE(tensor_.data.data, initial_data);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithEquivalentBufferSizeIsANoop) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  const void* const initial_data = tensor_.data.data;

  const std::vector<int> new_shape = {3, 4, 5};
  ASSERT_THAT(new_shape, Not(ElementsAreArray(reference_shape_)));
  TfLiteIntArray* dims2 = ConvertVectorToTfLiteIntArray(new_shape);
  ASSERT_EQ(BytesFor(type_, *dims2), reference_dims_bytes_);

  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims2), kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_EQ(tensor_.data.data, initial_data);
  EXPECT_EQ(tensor_.dims, dims2);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithDifferentShapeReallocatesData) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);

  TfLiteIntArray* dims2 = ConvertVectorToTfLiteIntArray({5, 4, 6});
  const int dims2_bytes = BytesFor(type_, *dims2);
  ASSERT_NE(dims2_bytes, reference_dims_bytes_);

  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims2), kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, dims2_bytes);
  EXPECT_EQ(tensor_.dims, dims2);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, dims2_bytes, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithSameShapeButDifferentBytesReallocatesData) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  // Resize the tensor manually with more bytes than it already has.
  TfLiteTensorResizeMaybeCopy(tensor_.bytes + 15, &tensor_,
                              /*preserve_data=*/true);
  ASSERT_GT(tensor_.bytes, reference_dims_bytes_);

  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_EQ(tensor_.dims, dims_);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, tensor_.bytes, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithSameShapeButStringTypeSizeReallocatesData) {
  constexpr size_t manual_bytes = 10;
  // Allocate the tensor manually.
  TfLiteTensorResizeMaybeCopy(manual_bytes, &tensor_, /*preserve_data=*/true);
  tensor_.dims = dims_;
  tensor_.type = kTfLiteString;

  // This fill fail if the memory reuse check cannot compute the new byte size.
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_EQ(tensor_.dims, dims_);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, tensor_.bytes, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithSameShapeButRessourceTypeSizeReallocatesData) {
  constexpr size_t manual_bytes = 10;
  // Allocate the tensor manually.
  TfLiteTensorResizeMaybeCopy(manual_bytes, &tensor_, /*preserve_data=*/true);
  tensor_.dims = dims_;
  tensor_.type = kTfLiteResource;

  // This fill fail if the memory reuse check cannot compute the new byte size.
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_EQ(tensor_.dims, dims_);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, tensor_.bytes, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithSameShapeButVariantTypeSizeReallocatesData) {
  constexpr size_t manual_bytes = 10;
  // Allocate the tensor manually.
  TfLiteTensorResizeMaybeCopy(manual_bytes, &tensor_, /*preserve_data=*/true);
  tensor_.dims = dims_;
  tensor_.type = kTfLiteVariant;

  // This fill fail if the memory reuse check cannot compute the new byte size.
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_EQ(tensor_.dims, dims_);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, tensor_.bytes, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST(ReplaceNodeSubsetsWithDelegateKernels,
     HandlesNodeSubsetWithNoOutputTensors) {
  static constexpr int kNumTensors = 1;
  static constexpr int kInputTensorIndex = 0;
  static constexpr int kExpectedPartitions = 1;
  static constexpr int kExpectedReplacedNodes = 1;
  static constexpr int kExpectedInputTensors = 1;
  static constexpr int kExpectedOutputTensors = 0;

  // Build a subgraph with a single node that has one input tensor and zero
  // output tensors (matching the shape of ASSIGN_VARIABLE).
  Interpreter interpreter;
  Subgraph& subgraph = interpreter.primary_subgraph();
  ASSERT_EQ(subgraph.AddTensors(/*tensors_to_add=*/kNumTensors), kTfLiteOk);
  ASSERT_EQ(subgraph.SetInputs({kInputTensorIndex}), kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({}), kTfLiteOk);

  TfLiteRegistration op_registration = {};
  int node_index = -1;
  ASSERT_EQ(
      subgraph.AddNodeWithParameters(
          /*inputs=*/{kInputTensorIndex}, /*outputs=*/{}, /*intermediates=*/{},
          /*init_data=*/nullptr, /*init_data_size=*/0,
          /*builtin_data=*/nullptr, &op_registration, &node_index),
      kTfLiteOk);

  // Delegate the partition and verify that both preview partitioning and
  // delegate kernel initialization receive output_tensors->size == 0 without
  // triggering undefined behavior in CopyVectorToTfLiteIntArray.
  bool kernel_initialized = false;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  delegate.data_ = &kernel_initialized;
  delegate.Prepare = [](TfLiteContext* context,
                        TfLiteDelegate* delegate) -> TfLiteStatus {
    TfLiteIntArray* execution_plan = nullptr;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return kTfLiteError;
    }

    TfLiteDelegateParams* preview_params = nullptr;
    int num_partitions = 0;
    if (context->PreviewDelegatePartitioning(context, execution_plan,
                                             &preview_params,
                                             &num_partitions) != kTfLiteOk) {
      return kTfLiteError;
    }
    EXPECT_EQ(num_partitions, kExpectedPartitions);
    if (preview_params && num_partitions == kExpectedPartitions) {
      EXPECT_EQ(preview_params[0].nodes_to_replace->size,
                kExpectedReplacedNodes);
      EXPECT_EQ(preview_params[0].input_tensors->size, kExpectedInputTensors);
      if (preview_params[0].input_tensors->size == kExpectedInputTensors) {
        EXPECT_EQ(preview_params[0].input_tensors->data[0], kInputTensorIndex);
      }
      EXPECT_EQ(preview_params[0].output_tensors->size, kExpectedOutputTensors);
    }

    TfLiteRegistration delegate_kernel_registration = {};
    delegate_kernel_registration.init =
        [](TfLiteContext* context, const char* buffer, size_t length) -> void* {
      const TfLiteDelegateParams* params =
          reinterpret_cast<const TfLiteDelegateParams*>(buffer);
      if (!params) {
        ADD_FAILURE() << "Expected non-null TfLiteDelegateParams.";
        return nullptr;
      }
      if (params->delegate && params->delegate->data_) {
        *static_cast<bool*>(params->delegate->data_) = true;
      }
      EXPECT_EQ(params->nodes_to_replace->size, kExpectedReplacedNodes);
      EXPECT_EQ(params->input_tensors->size, kExpectedInputTensors);
      if (params->input_tensors->size == kExpectedInputTensors) {
        EXPECT_EQ(params->input_tensors->data[0], kInputTensorIndex);
      }
      EXPECT_EQ(params->output_tensors->size, kExpectedOutputTensors);
      return nullptr;
    };

    return context->ReplaceNodeSubsetsWithDelegateKernels(
        context, delegate_kernel_registration, execution_plan, delegate);
  };

  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(&delegate), kTfLiteOk);
  EXPECT_TRUE(kernel_initialized);
}

TEST(ReplaceNodeWithSubgraph, PreservesOptionalTensorIndices) {
  static StderrReporter* error_reporter = new StderrReporter;
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 2; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/error_reporter,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }

  Subgraph& primary = *subgraphs[0];
  Subgraph& decomp = *subgraphs[1];

  // Primary subgraph has 2 tensors (input 0, output 1) and 1 composite node.
  ASSERT_EQ(primary.AddTensors(/*tensors_to_add=*/2), kTfLiteOk);
  ASSERT_EQ(primary.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(primary.SetOutputs({1}), kTfLiteOk);
  TfLiteRegistration composite_reg = {};
  int composite_node_index = -1;
  ASSERT_EQ(primary.AddNodeWithParameters(
                /*inputs=*/{0, kTfLiteOptionalTensor},
                /*outputs=*/{1, kTfLiteOptionalTensor},
                /*intermediates=*/{}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr, &composite_reg,
                &composite_node_index),
            kTfLiteOk);

  // Decomposition subgraph has 3 tensors: input 0, internal tensor 1, output 2.
  // Its op uses kTfLiteOptionalTensor (-1) in inputs, outputs, and
  // intermediates (e.g. FULLY_CONNECTED without a bias tensor).
  ASSERT_EQ(decomp.AddTensors(/*tensors_to_add=*/3), kTfLiteOk);
  ASSERT_EQ(decomp.SetInputs({0, kTfLiteOptionalTensor}), kTfLiteOk);
  ASSERT_EQ(decomp.SetOutputs({2, kTfLiteOptionalTensor}), kTfLiteOk);
  TfLiteRegistration inner_reg = {};
  int inner_node_index = -1;
  ASSERT_EQ(decomp.AddNodeWithParameters(
                /*inputs=*/{0, 1, kTfLiteOptionalTensor},
                /*outputs=*/{2, kTfLiteOptionalTensor},
                /*intermediates=*/{1, kTfLiteOptionalTensor},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &inner_reg, &inner_node_index),
            kTfLiteOk);

  const auto* composite_node_and_reg =
      primary.node_and_registration(composite_node_index);
  ASSERT_NE(composite_node_and_reg, nullptr);
  int last_inserted_execution_index = -1;
  ASSERT_EQ(primary.ReplaceNodeWithSubgraph(
                /*execution_index=*/0, composite_node_and_reg->first,
                /*subgraph_index=*/1, last_inserted_execution_index),
            kTfLiteOk);
  EXPECT_EQ(last_inserted_execution_index, 0);

  ASSERT_EQ(primary.execution_plan().size(), 1);
  const int inlined_node_index = primary.execution_plan()[0];
  const auto* inlined_node_and_reg =
      primary.node_and_registration(inlined_node_index);
  ASSERT_NE(inlined_node_and_reg, nullptr);
  const TfLiteNode& inlined_node = inlined_node_and_reg->first;

  // Input 0 maps to primary tensor 0, internal tensor 1 is offset by 2 -> 3,
  // and kTfLiteOptionalTensor (-1) is preserved rather than becoming 2 + (-1).
  ASSERT_NE(inlined_node.inputs, nullptr);
  EXPECT_THAT(
      std::vector<int>(inlined_node.inputs->data,
                       inlined_node.inputs->data + inlined_node.inputs->size),
      ElementsAre(0, 3, kTfLiteOptionalTensor));

  // Output 2 maps to primary tensor 1, and kTfLiteOptionalTensor is preserved.
  ASSERT_NE(inlined_node.outputs, nullptr);
  EXPECT_THAT(
      std::vector<int>(inlined_node.outputs->data,
                       inlined_node.outputs->data + inlined_node.outputs->size),
      ElementsAre(1, kTfLiteOptionalTensor));

  // Intermediate 1 is offset by 2 -> 3, and kTfLiteOptionalTensor is preserved.
  ASSERT_NE(inlined_node.intermediates, nullptr);
  EXPECT_THAT(std::vector<int>(inlined_node.intermediates->data,
                               inlined_node.intermediates->data +
                                   inlined_node.intermediates->size),
              ElementsAre(3, kTfLiteOptionalTensor));
}

TEST(SubgraphCustomAllocationTest,
     ShrunkCustomOutputAfterCompositeInliningWithDynamicOp) {
  StderrReporter error_reporter;
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 2; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/&error_reporter,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }
  Subgraph& primary = *subgraphs[0];
  Subgraph& decomp = *subgraphs[1];

  const TfLiteQuantization no_quant = {kTfLiteNoQuantization, nullptr};
  ASSERT_EQ(primary.AddTensors(/*tensors_to_add=*/3), kTfLiteOk);
  ASSERT_EQ(primary.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(primary.SetOutputs({2}), kTfLiteOk);
  ASSERT_EQ(primary.SetTensorParametersReadWrite(
                /*tensor_index=*/0, kTfLiteFloat32, "in", /*dims=*/{8},
                no_quant, /*is_variable=*/false, /*dims_signature=*/{-1}),
            kTfLiteOk);
  ASSERT_EQ(
      primary.SetTensorParametersReadWrite(
          /*tensor_index=*/1, kTfLiteFloat32, "mid", /*dims=*/{1}, no_quant),
      kTfLiteOk);
  ASSERT_EQ(primary.SetTensorParametersReadWrite(
                /*tensor_index=*/2, kTfLiteFloat32, "out", /*dims=*/{8},
                no_quant, /*is_variable=*/false, /*dims_signature=*/{-1}),
            kTfLiteOk);

  TfLiteRegistration composite_op = {};
  int composite_node_index = -1;
  ASSERT_EQ(primary.AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &composite_op, &composite_node_index),
            kTfLiteOk);
  ASSERT_EQ(composite_node_index, 0);

  // Node 1 copies input 0 to output 2 and resizes output 2 in Prepare().
  TfLiteRegistration copy_op = {};
  copy_op.prepare = [](TfLiteContext* context,
                       TfLiteNode* node) -> TfLiteStatus {
    const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    return context->ResizeTensor(context, output,
                                 TfLiteIntArrayCopy(input->dims));
  };
  copy_op.invoke = [](TfLiteContext* context,
                      TfLiteNode* node) -> TfLiteStatus {
    const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    std::memcpy(output->data.raw, input->data.raw, input->bytes);
    return kTfLiteOk;
  };
  int copy_node_index = -1;
  ASSERT_EQ(primary.AddNodeWithParameters(
                /*inputs=*/{0, 1}, /*outputs=*/{2}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &copy_op, &copy_node_index),
            kTfLiteOk);
  ASSERT_EQ(copy_node_index, 1);

  // Decomposition subgraph has a static op followed by a dynamic-output op.
  // Inlining it at execution index 0 produces execution_plan_ = {2, 3, 1}, so
  // PrepareOpsStartingAt() stops at execution-plan index 1 (node 3) before
  // preparing copy_op (node 1).
  ASSERT_EQ(decomp.AddTensors(/*tensors_to_add=*/3), kTfLiteOk);
  ASSERT_EQ(decomp.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(decomp.SetOutputs({2}), kTfLiteOk);
  ASSERT_EQ(decomp.SetTensorParametersReadWrite(
                /*tensor_index=*/0, kTfLiteFloat32, "d_in", /*dims=*/{8},
                no_quant, /*is_variable=*/false, /*dims_signature=*/{-1}),
            kTfLiteOk);
  ASSERT_EQ(
      decomp.SetTensorParametersReadWrite(
          /*tensor_index=*/1, kTfLiteFloat32, "d_tmp", /*dims=*/{1}, no_quant),
      kTfLiteOk);
  ASSERT_EQ(
      decomp.SetTensorParametersReadWrite(
          /*tensor_index=*/2, kTfLiteFloat32, "d_out", /*dims=*/{1}, no_quant),
      kTfLiteOk);

  TfLiteRegistration static_noop = {};
  static_noop.invoke = [](TfLiteContext* context,
                          TfLiteNode* node) -> TfLiteStatus {
    return kTfLiteOk;
  };
  ASSERT_EQ(decomp.AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &static_noop),
            kTfLiteOk);

  TfLiteRegistration dynamic_op = {};
  dynamic_op.prepare = [](TfLiteContext* context,
                          TfLiteNode* node) -> TfLiteStatus {
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    output->allocation_type = kTfLiteDynamic;
    return kTfLiteOk;
  };
  dynamic_op.invoke = [](TfLiteContext* context,
                         TfLiteNode* node) -> TfLiteStatus {
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    TfLiteIntArray* dims = TfLiteIntArrayCreate(1);
    dims->data[0] = 1;
    if (context->ResizeTensor(context, output, dims) != kTfLiteOk) {
      return kTfLiteError;
    }
    output->data.f[0] = 1.0f;
    return kTfLiteOk;
  };
  ASSERT_EQ(decomp.AddNodeWithParameters(
                /*inputs=*/{1}, /*outputs=*/{2}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &dynamic_op),
            kTfLiteOk);

  const auto* composite_node_and_reg =
      primary.node_and_registration(composite_node_index);
  ASSERT_NE(composite_node_and_reg, nullptr);
  int last_inserted_execution_index = -1;
  ASSERT_EQ(primary.ReplaceNodeWithSubgraph(
                /*execution_index=*/0, composite_node_and_reg->first,
                /*subgraph_index=*/1, last_inserted_execution_index),
            kTfLiteOk);
  ASSERT_THAT(primary.execution_plan(), ElementsAre(2, 3, 1));

  alignas(kDefaultTensorAlignment) std::array<float, 8> large_in = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  alignas(kDefaultTensorAlignment) std::array<float, 8> large_out = {};
  ASSERT_EQ(primary.SetCustomAllocationForTensor(
                0, {large_in.data(), large_in.size() * sizeof(float)}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetCustomAllocationForTensor(
                2, {large_out.data(), large_out.size() * sizeof(float)}),
            kTfLiteOk);
  ASSERT_EQ(primary.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(primary.Invoke(), kTfLiteOk);
  EXPECT_THAT(large_out,
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f));

  // Shrink the input and custom output allocation below the previous run's
  // size. Output 2's stale byte size must not be verified before copy_op
  // (node 1, at execution-plan index 2) is prepared during Invoke().
  primary.ClearCustomAllocations();
  ASSERT_EQ(primary.ResizeInputTensor(0, {2}), kTfLiteOk);

  alignas(kDefaultTensorAlignment) std::array<float, 2> small_in = {10.0f,
                                                                    20.0f};
  alignas(kDefaultTensorAlignment) std::array<float, 2> small_out = {};
  ASSERT_EQ(primary.SetCustomAllocationForTensor(
                0, {small_in.data(), small_in.size() * sizeof(float)}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetCustomAllocationForTensor(
                2, {small_out.data(), small_out.size() * sizeof(float)}),
            kTfLiteOk);
  ASSERT_EQ(primary.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(primary.Invoke(), kTfLiteOk);
  EXPECT_THAT(small_out, ElementsAre(10.0f, 20.0f));
}

TEST(SubgraphCustomAllocationTest,
     RejectsUndersizedCustomOutputOnDelegatedGraph) {
  Interpreter interpreter;
  Subgraph& subgraph = interpreter.primary_subgraph();
  ASSERT_EQ(subgraph.AddTensors(/*tensors_to_add=*/3), kTfLiteOk);
  ASSERT_EQ(subgraph.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(subgraph.SetOutputs({2}), kTfLiteOk);

  const TfLiteQuantization no_quant = {kTfLiteNoQuantization, nullptr};
  ASSERT_EQ(subgraph.SetTensorParametersReadWrite(
                /*tensor_index=*/0, kTfLiteFloat32, "in", /*dims=*/{4},
                no_quant, /*is_variable=*/false, /*dims_signature=*/{-1}),
            kTfLiteOk);
  ASSERT_EQ(
      subgraph.SetTensorParametersReadWrite(
          /*tensor_index=*/1, kTfLiteFloat32, "mid", /*dims=*/{4}, no_quant),
      kTfLiteOk);
  ASSERT_EQ(subgraph.SetTensorParametersReadWrite(
                /*tensor_index=*/2, kTfLiteFloat32, "out", /*dims=*/{4},
                no_quant, /*is_variable=*/false, /*dims_signature=*/{-1}),
            kTfLiteOk);

  TfLiteRegistration noop = {};
  ASSERT_EQ(subgraph.AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &noop),
            kTfLiteOk);
  ASSERT_EQ(subgraph.AddNodeWithParameters(
                /*inputs=*/{1}, /*outputs=*/{2}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &noop),
            kTfLiteOk);

  // Replace nodes {0, 1} with a single delegate kernel at node index 2, so
  // execution_plan_ is {2} while last_exec_plan_index_prepared is 0.
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  delegate.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  delegate.Prepare = [](TfLiteContext* context,
                        TfLiteDelegate* delegate) -> TfLiteStatus {
    TfLiteIntArray* execution_plan = nullptr;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return kTfLiteError;
    }
    TfLiteRegistration delegate_kernel = {};
    delegate_kernel.prepare = [](TfLiteContext* context,
                                 TfLiteNode* node) -> TfLiteStatus {
      const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
      return context->ResizeTensor(context, output,
                                   TfLiteIntArrayCopy(input->dims));
    };
    return context->ReplaceNodeSubsetsWithDelegateKernels(
        context, delegate_kernel, execution_plan, delegate);
  };
  ASSERT_EQ(interpreter.ModifyGraphWithDelegate(&delegate), kTfLiteOk);
  ASSERT_THAT(subgraph.execution_plan(), ElementsAre(2));

  // Grow input 0 from {4} to {8} while keeping output 2's custom buffer sized
  // for {4}, forcing AllocateTensors() through PrepareOpsAndTensors().
  ASSERT_EQ(subgraph.ResizeInputTensor(0, {8}), kTfLiteOk);

  alignas(kDefaultTensorAlignment) std::array<float, 8> in_buf = {};
  alignas(kDefaultTensorAlignment) std::array<float, 4> undersized_out_buf = {};
  ASSERT_EQ(subgraph.SetCustomAllocationForTensor(
                0, {in_buf.data(), in_buf.size() * sizeof(float)}),
            kTfLiteOk);
  ASSERT_EQ(subgraph.SetCustomAllocationForTensor(
                2, {undersized_out_buf.data(),
                    undersized_out_buf.size() * sizeof(float)}),
            kTfLiteOk);

  EXPECT_EQ(subgraph.AllocateTensors(), kTfLiteError);
}

TEST(SubgraphCustomAllocationTest,
     HandlesEmptyExecutionPlanAfterInliningEmptySubgraph) {
  StderrReporter error_reporter;
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 2; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/&error_reporter,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }
  Subgraph& primary = *subgraphs[0];
  Subgraph& decomp = *subgraphs[1];

  const TfLiteQuantization no_quant = {kTfLiteNoQuantization, nullptr};
  ASSERT_EQ(primary.AddTensors(/*tensors_to_add=*/2), kTfLiteOk);
  ASSERT_EQ(primary.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(primary.SetOutputs({0}), kTfLiteOk);
  ASSERT_EQ(
      primary.SetTensorParametersReadWrite(
          /*tensor_index=*/0, kTfLiteFloat32, "in", /*dims=*/{4}, no_quant),
      kTfLiteOk);
  ASSERT_EQ(primary.SetTensorParametersReadWrite(
                /*tensor_index=*/1, kTfLiteFloat32, "dead_out", /*dims=*/{4},
                no_quant),
            kTfLiteOk);

  TfLiteRegistration composite_op = {};
  int composite_node_index = -1;
  ASSERT_EQ(primary.AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*intermediates=*/{},
                /*init_data=*/nullptr, /*init_data_size=*/0,
                /*builtin_data=*/nullptr, &composite_op, &composite_node_index),
            kTfLiteOk);

  ASSERT_EQ(decomp.AddTensors(/*tensors_to_add=*/2), kTfLiteOk);
  ASSERT_EQ(decomp.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(decomp.SetOutputs({1}), kTfLiteOk);

  const auto* composite_node_and_reg =
      primary.node_and_registration(composite_node_index);
  ASSERT_NE(composite_node_and_reg, nullptr);
  int last_inserted_execution_index = -1;
  ASSERT_EQ(primary.ReplaceNodeWithSubgraph(
                /*execution_index=*/0, composite_node_and_reg->first,
                /*subgraph_index=*/1, last_inserted_execution_index),
            kTfLiteOk);
  ASSERT_TRUE(primary.execution_plan().empty());
  ASSERT_EQ(primary.nodes_size(), 1);

  alignas(kDefaultTensorAlignment) std::array<float, 4> in_buf = {};
  alignas(kDefaultTensorAlignment) std::array<float, 1> dead_out_buf = {};
  ASSERT_EQ(primary.SetCustomAllocationForTensor(
                0, {in_buf.data(), in_buf.size() * sizeof(float)}),
            kTfLiteOk);
  ASSERT_EQ(primary.SetCustomAllocationForTensor(
                1, {dead_out_buf.data(), dead_out_buf.size() * sizeof(float)}),
            kTfLiteOk);

  EXPECT_EQ(primary.AllocateTensors(), kTfLiteOk);
}

}  // namespace
}  // namespace tflite
