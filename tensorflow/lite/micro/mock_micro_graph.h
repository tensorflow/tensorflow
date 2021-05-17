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

#ifndef TENSORFLOW_LITE_MICRO_MOCK_MICRO_GRAPH_H_
#define TENSORFLOW_LITE_MICRO_MOCK_MICRO_GRAPH_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// MockMicroGraph stubs out all MicroGraph methods used during invoke. A count
// of the number of calls to invoke for each subgraph is maintained for
// validation of control flow operators.
class MockMicroGraph : public MicroGraph {
 public:
  explicit MockMicroGraph(SimpleMemoryAllocator* allocator);
  TfLiteStatus InvokeSubgraph(int subgraph_idx) override;
  TfLiteStatus ResetVariableTensors() override;
  size_t NumSubgraphInputs(int subgraph_idx) override;
  TfLiteEvalTensor* GetSubgraphInput(int subgraph_idx, int tensor_idx) override;
  size_t NumSubgraphOutputs(int subgraph_idx) override;
  TfLiteEvalTensor* GetSubgraphOutput(int subgraph_idx,
                                      int tensor_idx) override;
  int NumSubgraphs() override;
  int get_init_count() const { return init_count_; }
  int get_prepare_count() const { return prepare_count_; }
  int get_free_count() const { return free_count_; }
  int get_invoke_count(int subgraph_idx) const {
    return invoke_counts_[subgraph_idx];
  }

 private:
  static constexpr int kMaxSubgraphs = 10;
  SimpleMemoryAllocator* allocator_;
  TfLiteEvalTensor* mock_tensor_;
  int init_count_;
  int prepare_count_;
  int free_count_;
  int invoke_counts_[kMaxSubgraphs];
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MOCK_MICRO_GRAPH_H_
