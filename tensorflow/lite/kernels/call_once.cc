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

#include <stddef.h>

#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/initialization_status.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace call_once_kernel {

// CallOnce operator is a control flow op to invoke other subgraph in the graph
// in order to conduct the given graph's initialization tasks, for example, hash
// table initialization and variable initialization.
//
// This operator will invoke the subgraph for initialization in the first run
// and become no-op after the first run in an interpreter's life cycle.

struct OpData {
  // Subgraph index to be invoked once in a life cycle by this CallOnce op.
  int init_subgraph_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteCallOnceParams*>(buffer);
  op_data->init_subgraph_index = params->init_subgraph_index;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  // The CALL_ONCE op cannot call the same subgraph recursively or it will cause
  // an infinite recursion.
  TF_LITE_ENSURE(context, op_data->init_subgraph_index !=
                              this_subgraph->GetSubgraphIndex());

  // Return early if the initialization graph is already invoked.
  resource::InitializationStatusMap* map =
      &this_subgraph->initialization_status_map();
  resource::InitializationStatus* status =
      resource::GetInitializationStatus(map, op_data->init_subgraph_index);
  if (status->IsInitialized()) return kTfLiteOk;

  auto* subgraphs = this_subgraph->GetSubgraphs();

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 0);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 0);

  TF_LITE_ENSURE(context, op_data->init_subgraph_index < subgraphs->size());

  // Ensures that there are no input and output tensors in the subgraph.
  Subgraph* init_subgraph = (*subgraphs)[op_data->init_subgraph_index].get();
  TF_LITE_ENSURE_EQ(context, init_subgraph->inputs().size(), 0);
  TF_LITE_ENSURE_EQ(context, init_subgraph->outputs().size(), 0);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  // The initialization graph should be invoked once in a life cycle.
  resource::InitializationStatusMap* map =
      &this_subgraph->initialization_status_map();
  resource::InitializationStatus* status =
      resource::GetInitializationStatus(map, op_data->init_subgraph_index);
  if (status->IsInitialized()) return kTfLiteOk;

  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph& init_subgraph = *(*subgraphs)[op_data->init_subgraph_index];

  TF_LITE_ENSURE_OK(context, init_subgraph.AllocateTensors());
  TF_LITE_ENSURE_OK(context, init_subgraph.Invoke());
  TF_LITE_ENSURE_OK(context, init_subgraph.ReleaseNonPersistentMemory());

  // Mark the invocation completed.
  status->MarkInitializationIsDone();
  return kTfLiteOk;
}

}  // namespace call_once_kernel

TfLiteRegistration* Register_CALL_ONCE() {
  static TfLiteRegistration r = {call_once_kernel::Init, call_once_kernel::Free,
                                 call_once_kernel::Prepare,
                                 call_once_kernel::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
