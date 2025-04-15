/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_SAMPLE_STABLE_DELEGATE_SAMPLE_STABLE_DELEGATE_WITH_CONTROL_FLOW_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_SAMPLE_STABLE_DELEGATE_SAMPLE_STABLE_DELEGATE_WITH_CONTROL_FLOW_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace example {
namespace helpers {
int CalculateNumElements(const TfLiteOpaqueTensor* opaque_tensor);
}  // namespace helpers

// LINT.IfChange
static const char kSampleStableDelegateName[] = "google_sample_delegate";
// LINT.ThenChange(Google-internal path)
static const char kSampleStableDelegateVersion[] = "1.0.0";

// A simple delegate that supports only a few operations:
// addition, subtraction, multiplication, equality checks, and while loops.
// Implements SimpleOpaqueDelegateInterface, and therefore the delegate can be
// easily be adapted to work with the stable TFLite delegate API via
// TfLiteOpaqueDelegateFactory.
class SampleStableDelegate : public SimpleOpaqueDelegateInterface {
 public:
  // SampleStableDelegate supports float32 input type only.
  // Returns true if the inputs of 'node' are two tensors of float32 with the
  // same shape and the operation is supported (without fused activation).
  bool IsNodeSupportedByDelegate(const TfLiteOperator* registration_external,
                                 const TfLiteOpaqueNode* node,
                                 TfLiteOpaqueContext* context) const override;

  // No-op. The delegate doesn't have extra steps to perform during
  // initialization.
  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override;

  // Returns a name that identifies the delegate.
  const char* Name() const override;

  // Returns an instance of SampleStableDelegateKernel that implements
  // SimpleOpaqueDelegateKernelInterface. SampleStableDelegateKernel describes
  // how a subgraph is delegated and the concrete evaluation of operations to be
  // performed by the delegate.
  std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override;

 private:
  // Computes all the compatible callee subgraphs of control flow ops of the
  // subgraph specified with the given index. All the subgraph tree structures
  // are stored in control_flow_subgraph_tree_ and any compatible subgraphs are
  // added to compatible_callee_subgraph_indices_.
  // NOTE: This function is expected to be called recursively to gather all the
  // nested control flow subgraphs, and is expected to be called by
  // PrepareControlFlow() with the root (subgraph_index = 0).
  TfLiteStatus ComputeCompatibleCalleeSubgraphs(
      TfLiteOpaqueContext* opaque_context, int subgraph_index);
  // Performs any necessary steps for control flow support. For this sample
  // delegate, we computes compatible callee subgraphs, releases subgraph
  // contexts, and mark compatible callee subgraphs so that we can avoid Calling
  // ModifyGraphWithDelegate() on the compatible subgraphs.
  TfLiteStatus PrepareControlFlow(TfLiteOpaqueContext* opaque_context);

  // Adds a control flow callee subgraph to the parent subgraph in the
  // control_flow_subgraph_tree_.
  void AddCalleeSubgraphToCallerSubgraph(int callee_subgraph_index,
                                         int caller_subgraph_index) {
    control_flow_subgraph_tree_[caller_subgraph_index].insert(
        callee_subgraph_index);
  }

  // Adds a compatible callee subgraph.
  void AddCompatibleCalleeSubgraph(int subgraph_index) {
    compatible_callee_subgraph_indices_.insert(subgraph_index);
  }

  // Returns true if `subgraph_index` is of a compatible callee subgraph.
  bool IsCompatibleCalleeSubgraph(int subgraph_index) const {
    return compatible_callee_subgraph_indices_.contains(subgraph_index);
  }
  // A map from a parent subgraph index to its control flow callee subgraph
  // indices (i.e. called by WHILE op). This information is used for
  // constructing a tree of control flow subgraphs and traversing to figure out
  // the delegation dependencies.
  absl::flat_hash_map<int, absl::flat_hash_set<int>>
      control_flow_subgraph_tree_;
  // A set of callee subgraph indices (i.e., called by WHILE op) that this
  // sample delegate can fully support. We mark all the callee subgraphs of the
  // subgraph S (that contains this control flow op) as compatible if all those
  // callee subgraphs are fully supported (i.e. contains only the ops fully
  // supported by this delegate). For example, a WHILE op contains two callees:
  // condition and body subgraphs. If and only if both condition and body
  // subgraphs contain only the supported ops, then both subgraphs are marked
  // as compatible.
  // NOTE: The definition of `compatible` depends on the delegate provider's
  // requirements. The definition we provide in this sample delegate is just an
  // example for demonstration purpose only.
  absl::flat_hash_set<int> compatible_callee_subgraph_indices_;
  // If the delegate is already initialized. This is used to avoid duplicate
  // PrepareControlFlow() call (i.e. we only want to call PrepareControlFlow()
  // in the primary subgraph for our sample delegate).
  bool has_been_initialized_ = false;
};

}  // namespace example
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_EXPERIMENTAL_SAMPLE_STABLE_DELEGATE_SAMPLE_STABLE_DELEGATE_WITH_CONTROL_FLOW_H_
