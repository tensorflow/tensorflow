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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_GRAPH_H_
#define TENSORFLOW_LITE_MICRO_MICRO_GRAPH_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Abstracts the details of interacting with the tflite::Model.
//
// Provides methods to access, initialize, prepare, invoke and free any
// subgraph in the tflite::Graph.
class MicroGraph {
 public:
  // The lifetime of the context, model, and allocator must be at least as long
  // as that of the graph object, since the this class may need to access them
  // at any time.
  MicroGraph(TfLiteContext* context, const Model* model,
             MicroAllocator* allocator);
  virtual ~MicroGraph();

  // Sets up builtin data and calls TfLiteRegistration->Init for every operator
  // in every subgraph in the model.
  virtual TfLiteStatus InitSubgraphs();

  // Calls TfLiteRegistration->Prepare for every operator in every subgraph in
  // the model.
  virtual TfLiteStatus PrepareSubgraphs();

  // Calls TfLiteRegistration->Free for every operator in every subgraph in the
  // model.
  virtual TfLiteStatus FreeSubgraphs();

  // Calls TfLiteRegistration->Invoke for every operator in a single subgraph in
  // the model.
  virtual TfLiteStatus InvokeSubgraph(int subgraph_idx);

  // Zeros out all variable tensors in all subgraphs in the model.
  virtual TfLiteStatus ResetVariableTensors();

  // Number of tensor inputs to a specified subgraph in the model.
  virtual size_t NumSubgraphInputs(int subgraph_idx);

  // Get the specified input tensor of a specified subgraph in the model.
  virtual TfLiteEvalTensor* GetSubgraphInput(int subgraph_idx, int input_idx);

  // Number of tensor outputs to a specified subgraph in the model.
  virtual size_t NumSubgraphOutputs(int subgraph_idx);

  // Get the specified output tensor of a specified subgraph in the model.
  virtual TfLiteEvalTensor* GetSubgraphOutput(int subgraph_idx, int output_idx);

  // Number of subgraphs in the model.
  virtual int NumSubgraphs();

  // Hook to pass in subgraph allocations tracked within the interpreter,
  // allowing MicroGraph to init / prepare / invoke subgraphs in the model.
  void SetSubgraphAllocations(SubgraphAllocations* subgraph_allocations);

  // Get the current subgraph index. Within an on operator, this is guaranteed
  // to be the subgraph of that operator.
  int GetCurrentSubgraphIndex() { return current_subgraph_index_; }

  // Gets the list of alloctions for each subgraph. This is the source of truth
  // for all per-subgraph allocation data.
  SubgraphAllocations* GetAllocations() { return subgraph_allocations_; }

 private:
  TfLiteContext* context_;
  const Model* model_;
  MicroAllocator* allocator_;
  SubgraphAllocations* subgraph_allocations_ = nullptr;
  int current_subgraph_index_;
  const flatbuffers::Vector<flatbuffers::Offset<SubGraph>>* subgraphs_;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_GRAPH_H_
