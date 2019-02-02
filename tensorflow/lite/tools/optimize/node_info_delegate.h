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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_NODE_INFO_DELEGATE_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_NODE_INFO_DELEGATE_H_

#include <unordered_map>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/tools/optimize/calibration_common.h"

namespace tflite {
namespace optimize {
namespace calibration {

// An interface for delegate observer that can listen to TfLiteDelegate::Prepare
// calls.
class DelegateObserver {
 public:
  virtual TfLiteStatus OnDelegatePrepareCalled(TfLiteContext* context) = 0;
  virtual ~DelegateObserver() {}
};

// The parameters for the node info delegate.
struct NodeInfoDelegateParams {
  DelegateObserver* delegate_observer;
};

// Creates a delegate with the given |params|.
TfLiteDelegate CreateNodeInfoDelegate(NodeInfoDelegateParams* params);

// A delegate observer that can construct the map from TfLiteNode* ->
// OperatorInfo.
class NodeInfoDelegateObserver : public DelegateObserver {
 public:
  NodeInfoDelegateObserver(
      const std::unordered_map<int, OperatorInfo>& node_index_to_op,
      std::unordered_map<const TfLiteNode*, OperatorInfo>* node_ptr_opinfo_map)
      : node_index_opinfo_map_(node_index_to_op),
        node_ptr_opinfo_map_(node_ptr_opinfo_map) {}

  TfLiteStatus OnDelegatePrepareCalled(TfLiteContext* context) override;

  // Returns the context that was used to called the prepare method.
  const TfLiteContext* GetContext() const { return context_; }

 private:
  const TfLiteContext* context_ = nullptr;
  const std::unordered_map<int, OperatorInfo>& node_index_opinfo_map_;
  std::unordered_map<const TfLiteNode*, OperatorInfo>* node_ptr_opinfo_map_;
};

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_NODE_INFO_DELEGATE_H_
