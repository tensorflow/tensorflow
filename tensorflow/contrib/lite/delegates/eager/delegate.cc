/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/delegates/eager/delegate.h"

#include <vector>

#include "tensorflow/contrib/lite/context_util.h"
#include "tensorflow/contrib/lite/delegates/eager/buffer_map.h"
#include "tensorflow/contrib/lite/delegates/eager/kernel.h"
#include "tensorflow/contrib/lite/delegates/eager/util.h"
#include "tensorflow/contrib/lite/util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tflite {
namespace eager {
namespace delegate {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // Get the nodes in the current execution plan. Interpreter owns this array.
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  // Add all custom ops starting with "Eager" to list of supported nodes.
  std::vector<int> supported_nodes;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TfLiteNode* node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));

    if (IsEagerOp(registration->custom_name)) {
      supported_nodes.push_back(node_index);
    }
  }

  // Request TFLite to partition the graph and make kernels for each independent
  // subgraph.
  TfLiteIntArray* size_and_nodes =
      ConvertVectorToTfLiteIntArray(supported_nodes);
  context->ReplaceSubgraphsWithDelegateKernels(context, GetKernel(),
                                               size_and_nodes, delegate);
  TfLiteIntArrayFree(size_and_nodes);
  return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle, void* data,
                                  size_t size) {
  // TODO(nupurgarg): Make BufferMap unique to each interpreter in order to
  // support multiple interpreters using a single delegate.
  BufferMap* buffer_map =
      reinterpret_cast<DelegateData*>(delegate->data_)->GetBufferMap();

  // TODO(nupurgarg): Use TfLiteContext's ReportError instead of fprinf.
  if (!buffer_map->HasTensor(buffer_handle)) {
    fprintf(stderr, "Invalid tensor index %d.\n", buffer_handle);
    return kTfLiteError;
  }

  tensorflow::Tensor t = buffer_map->GetTensor(buffer_handle);
  tensorflow::StringPiece t_data = t.tensor_data();

  if (size != t_data.size()) {
    fprintf(stderr, "Not enough space to store TensorFlow's aligned buffer.\n");
    return kTfLiteError;
  }

  memcpy(data, t_data.data(), t_data.size());
  return kTfLiteOk;
}

}  // namespace delegate
}  // namespace eager

EagerDelegate::EagerDelegate() {}

EagerDelegate::~EagerDelegate() {}

TfLiteStatus EagerDelegate::Apply(Interpreter* interpreter) {
  if (!delegate_) {
    if (!eager::DelegateData::Create(&delegate_data_).ok()) {
      fprintf(stderr, "Unable to initialize TensorFlow context.\n");
      return kTfLiteError;
    }

    delegate_.reset(new TfLiteDelegate{
        /*data_=*/delegate_data_.get(),
        /*nullptr,*/ &eager::delegate::Prepare,
        /*CopyFromBufferHandle=*/&eager::delegate::CopyFromBufferHandle,
        /*CopyToBufferHandle=*/nullptr,
        /*FreeBufferHandle=*/nullptr});
  }

  return interpreter->ModifyGraphWithDelegate(delegate_.get(),
                                              /*allow_dynamic_tensors=*/true);
}

}  // namespace tflite
