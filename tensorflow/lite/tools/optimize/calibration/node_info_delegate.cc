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
#include "tensorflow/lite/tools/optimize/calibration/node_info_delegate.h"

namespace tflite {
namespace optimize {
namespace calibration {

namespace {
// The prepare function for delegate that forwards the prepare call to the
// delegate observer in node info delegate params.
// The function simply calls a delegate observer OnDelegatePrepareMethod.
TfLiteStatus NodeInfoDelegatePrepare(TfLiteContext* context,
                                     TfLiteDelegate* delegate) {
  if (delegate == nullptr) return TfLiteStatus::kTfLiteError;

  NodeInfoDelegateParams* params =
      reinterpret_cast<NodeInfoDelegateParams*>(delegate->data_);
  return params->delegate_observer->OnDelegatePrepareCalled(context);
}
}  // namespace

TfLiteDelegate CreateNodeInfoDelegate(NodeInfoDelegateParams* params) {
  return {/*data_ */ params,
          /* Prepare */ NodeInfoDelegatePrepare,
          /* CopyFromBufferHandle*/ nullptr,
          /* CopyToBufferHandle*/ nullptr,
          /* FreeBufferHandle*/ nullptr};
}

TfLiteStatus NodeInfoDelegateObserver::OnDelegatePrepareCalled(
    TfLiteContext* context) {
  context_ = context;
  const size_t num_nodes = node_index_opinfo_map_.size();
  for (size_t node_index = 0; node_index < num_nodes; node_index++) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* reg = nullptr;
    TF_LITE_ENSURE_STATUS(
        context->GetNodeAndRegistration(context, node_index, &node, &reg));
    auto op_info = node_index_opinfo_map_.at(node_index);
    op_info.registration = reg;
    node_ptr_opinfo_map_->insert({node, op_info});
  }

  if (node_ptr_opinfo_map_->size() != node_index_opinfo_map_.size()) {
    // Something wrong.
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
