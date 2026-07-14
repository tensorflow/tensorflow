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
#include "tensorflow/lite/delegates/utils.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace delegates {
namespace {

TEST(UtilsTest, CreateNewTensorWithDifferentTypeTest) {
  std::vector<TfLiteTensor> tensors(2);
  // Data about original tensor.
  // The same shape should be reflected in tensors[1] later.
  tensors[0].dims = TfLiteIntArrayCreate(2);
  tensors[0].dims->data[0] = 2;
  tensors[0].dims->data[1] = 3;
  tensors[0].type = kTfLiteFloat32;
  // To simulate a valid TFLite Context.
  TfLiteContext context;
  context.AddTensors = [](struct TfLiteContext*, int tensors_to_add,
                          int* first_new_tensor_index) {
    // The util should be adding exactly one tensor to the graph.
    if (tensors_to_add != 1) {
      return kTfLiteError;
    }
    // This ensures that the 'new tensor' is the second tensor in the vector
    // above.
    *first_new_tensor_index = 1;
    return kTfLiteOk;
  };
  context.ResizeTensor = [](struct TfLiteContext*, TfLiteTensor* tensor,
                            TfLiteIntArray* new_size) {
    // Ensure dimensions are the same as the original tensor.
    if (new_size->size != 2 || new_size->data[0] != 2 || new_size->data[1] != 3)
      return kTfLiteError;
    tensor->dims = new_size;
    return kTfLiteOk;
  };
  context.tensors = tensors.data();

  TfLiteTensor* new_tensor = nullptr;
  int new_tensor_index = -1;
  EXPECT_EQ(CreateNewTensorWithDifferentType(
                &context, /**original_tensor_index**/ 0,
                /**new_type**/ kTfLiteUInt8, &new_tensor, &new_tensor_index),
            kTfLiteOk);
  EXPECT_EQ(new_tensor_index, 1);
  EXPECT_NE(new_tensor, nullptr);
  EXPECT_NE(new_tensor->dims, nullptr);
  EXPECT_EQ(new_tensor->type, kTfLiteUInt8);
  EXPECT_EQ(new_tensor->allocation_type, kTfLiteArenaRw);

  // Cleanup.
  TfLiteIntArrayFree(tensors[0].dims);
  TfLiteIntArrayFree(tensors[1].dims);
}

TEST(UtilsTest, AcquireAndReleaseSubgraphContextTest) {
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 5; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/nullptr,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }
  TfLiteContext* context = subgraphs[0]->context();
  subgraphs[0]->context();

  TfLiteContext* acquired_context;
  EXPECT_EQ(kTfLiteOk, AcquireSubgraphContext(context, 2, &acquired_context));
  EXPECT_EQ(subgraphs[2]->context(), acquired_context);
  EXPECT_EQ(kTfLiteOk, ReleaseSubgraphContext(context, 2));
}

TEST(UtilsTest, MarkSubgraphAsDelegationSkippableTest) {
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 3; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/nullptr,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }
  TfLiteContext* context = subgraphs[0]->context();

  EXPECT_EQ(kTfLiteOk, MarkSubgraphAsDelegationSkippable(context, 2));

  EXPECT_FALSE(subgraphs[0]->IsDelegationSkippable());
  EXPECT_FALSE(subgraphs[1]->IsDelegationSkippable());
  EXPECT_TRUE(subgraphs[2]->IsDelegationSkippable());
}

// A mock TfLiteContext to be used for GraphPartitionHelperTest.
class MockTfLiteContext : public TfLiteContext {
 public:
  MockTfLiteContext() : TfLiteContext({0}) {
    // Simply create a 10-node execution plan.
    exec_plan_ = TfLiteIntArrayCreate(10);
    for (int i = 0; i < 10; ++i) exec_plan_->data[i] = i;

    // Create {1}, {0,3,7,8}, {2,4,9}, {5,6} 4 partitions.
    TfLiteDelegateParams params1({nullptr});
    params1.nodes_to_replace = TfLiteIntArrayCreate(1);
    params1.nodes_to_replace->data[0] = 1;
    delegate_params_.emplace_back(params1);

    TfLiteDelegateParams params2({nullptr});
    params2.nodes_to_replace = TfLiteIntArrayCreate(4);
    params2.nodes_to_replace->data[0] = 0;
    params2.nodes_to_replace->data[1] = 3;
    params2.nodes_to_replace->data[2] = 7;
    params2.nodes_to_replace->data[3] = 8;
    delegate_params_.emplace_back(params2);

    TfLiteDelegateParams params3({nullptr});
    params3.nodes_to_replace = TfLiteIntArrayCreate(3);
    params3.nodes_to_replace->data[0] = 2;
    params3.nodes_to_replace->data[1] = 4;
    params3.nodes_to_replace->data[2] = 9;
    delegate_params_.emplace_back(params3);

    TfLiteDelegateParams params4({nullptr});
    params4.nodes_to_replace = TfLiteIntArrayCreate(2);
    params4.nodes_to_replace->data[0] = 5;
    params4.nodes_to_replace->data[1] = 6;
    delegate_params_.emplace_back(params4);

    // We need to mock the following 3 functions inside TfLiteContext object
    // that are used by GraphPartitionHelper implementation.
    this->GetExecutionPlan = MockGetExecutionPlan;
    this->GetNodeAndRegistration = MockGetNodeAndRegistration;
    this->PreviewDelegatePartitioning = MockPreviewDelegatePartitioning;
  }
  ~MockTfLiteContext() {
    TfLiteIntArrayFree(exec_plan_);
    for (auto params : delegate_params_) {
      TfLiteIntArrayFree(params.nodes_to_replace);
      TfLiteIntArrayFree(params.input_tensors);
      TfLiteIntArrayFree(params.output_tensors);
    }
  }

  TfLiteIntArray* exec_plan() const { return exec_plan_; }
  TfLiteNode* node() { return &node_; }
  TfLiteRegistration* registration() { return &registration_; }
  TfLiteDelegateParams* delegate_params() { return &delegate_params_.front(); }
  int num_delegate_params() { return delegate_params_.size(); }

 private:
  static TfLiteStatus MockGetExecutionPlan(TfLiteContext* context,
                                           TfLiteIntArray** execution_plan) {
    MockTfLiteContext* mock = reinterpret_cast<MockTfLiteContext*>(context);
    *execution_plan = mock->exec_plan();
    return kTfLiteOk;
  }

  static TfLiteStatus MockGetNodeAndRegistration(
      TfLiteContext* context, int node_index, TfLiteNode** node,
      TfLiteRegistration** registration) {
    MockTfLiteContext* mock = reinterpret_cast<MockTfLiteContext*>(context);
    *node = mock->node();
    *registration = mock->registration();
    return kTfLiteOk;
  }

  static TfLiteStatus MockPreviewDelegatePartitioning(
      TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegateParams** partition_params_array, int* num_partitions) {
    MockTfLiteContext* mock = reinterpret_cast<MockTfLiteContext*>(context);
    *partition_params_array = mock->delegate_params();
    *num_partitions = mock->num_delegate_params();
    return kTfLiteOk;
  }

  // The execution plan of this mocked TfLiteContext object.
  TfLiteIntArray* exec_plan_;

  // For simplicity, the mocked graph has only type of node and one
  // registration.
  TfLiteNode node_;
  TfLiteRegistration registration_{};

  // The TfLiteDelegateParams object that's manually populated inside the mocked
  // TfLiteContext::PreviewDelegatePartitioning.
  std::vector<TfLiteDelegateParams> delegate_params_;
};

bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                     TfLiteRegistration* registration,
                     std::string* unsupported_details) {
  return true;
}

std::vector<int> GetNodesToReplaceFromPartitions(
    const std::vector<TfLiteDelegateParams*>& partitions) {
  std::vector<int> nodes;
  for (const auto p : partitions) {
    nodes.insert(nodes.end(), p->nodes_to_replace->data,
                 p->nodes_to_replace->data + p->nodes_to_replace->size);
  }
  return nodes;
}

TEST(GraphPartitionHelper, CheckPartitions) {
  // The mocked TfLiteContext has 4 partitions: {1}, {0,3,7,8}, {2,4,9}, {5,6}.
  MockTfLiteContext mocked_context;
  GraphPartitionHelper helper(&mocked_context, IsNodeSupported);
  EXPECT_EQ(kTfLiteOk, helper.Partition(nullptr));
  EXPECT_EQ(10, helper.num_total_nodes());
  EXPECT_EQ(4, helper.num_partitions());

  auto partitions = helper.GetFirstNLargestPartitions(1, 0);
  EXPECT_EQ(1, partitions.size());
  auto nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8}));

  // Get the largest partition but requiring at least 5 nodes, so empty result.
  partitions = helper.GetFirstNLargestPartitions(1, 5);
  EXPECT_TRUE(partitions.empty());

  partitions = helper.GetFirstNLargestPartitions(10, 3);
  EXPECT_EQ(2, partitions.size());
  EXPECT_EQ(4, partitions[0]->nodes_to_replace->size);
  EXPECT_EQ(3, partitions[1]->nodes_to_replace->size);
  nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8, 2, 4, 9}));
}

TfLiteStatus ErrorGetExecutionPlan(TfLiteContext* context,
                                   TfLiteIntArray** execution_plan) {
  return kTfLiteError;
}

void EmptyReportError(TfLiteContext* context, const char* format, ...) {}

TEST(GraphPartitionHelper, CheckPrepareErrors) {
  TfLiteContext error_context({0});
  error_context.GetExecutionPlan = ErrorGetExecutionPlan;
  error_context.ReportError = EmptyReportError;
  GraphPartitionHelper helper(&error_context, IsNodeSupported);
  EXPECT_EQ(kTfLiteError, helper.Partition(nullptr));
}

TEST(GraphPartitionHelper, CheckPartitionsWithSupportedNodeList) {
  // The mocked TfLiteContext has 4 partitions: {1}, {0,3,7,8}, {2,4,9}, {5,6}.
  // So, we simply create a list of supported nodes as {0,1,2,...,8,9}
  MockTfLiteContext mocked_context;
  std::vector<int> supported_nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  GraphPartitionHelper helper(&mocked_context, supported_nodes);
  EXPECT_EQ(kTfLiteOk, helper.Partition(nullptr));
  EXPECT_EQ(10, helper.num_total_nodes());
  EXPECT_EQ(4, helper.num_partitions());

  auto partitions = helper.GetFirstNLargestPartitions(1, 0);
  EXPECT_EQ(1, partitions.size());
  auto nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8}));

  // Get the largest partition but requiring at least 5 nodes, so empty result.
  partitions = helper.GetFirstNLargestPartitions(1, 5);
  EXPECT_TRUE(partitions.empty());

  partitions = helper.GetFirstNLargestPartitions(10, 3);
  EXPECT_EQ(2, partitions.size());
  EXPECT_EQ(4, partitions[0]->nodes_to_replace->size);
  EXPECT_EQ(3, partitions[1]->nodes_to_replace->size);
  nodes = GetNodesToReplaceFromPartitions(partitions);
  EXPECT_THAT(nodes, testing::ElementsAreArray({0, 3, 7, 8, 2, 4, 9}));
}

}  // namespace
}  // namespace delegates
}  // namespace tflite
