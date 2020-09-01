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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_TRANSFORMER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_TRANSFORMER_H_

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"

namespace tflite {
namespace gpu {

class TransformationReporter;

struct TransformationContext {
  GraphFloat32* graph;
  TransformationReporter* reporter;
};

enum class TransformStatus {
  // Transformation was not applied due to trivial conditions mismatch.
  //
  // This is different from DECLINED code below that provides in-depth
  // explanation why a transformation that could have been applied but was not
  // due to some issues.
  SKIPPED,

  // Transformation was declined, therefore, a model was not modified.
  DECLINED,

  // Transformation was applied successfully
  APPLIED,

  // Transformation may partially be applied, but left a model in an invalid
  // state. This error should be considered unrecoverable.
  INVALID,
};

struct TransformResult {
  TransformStatus status;
  std::string message;
};

// Class responsible for applying a transformation to a single node.
class NodeTransformation {
 public:
  virtual ~NodeTransformation() = default;

  virtual TransformResult ApplyToNode(Node* node, GraphFloat32* graph) = 0;
};

// Class responsible for applying a transformation to a sequence of nodes.
// Nodes are guaranteed to depend on each other without extra dependents being
// spilled.
class SequenceTransformation {
 public:
  virtual ~SequenceTransformation() = default;

  // @return number of nodes in a sequence to apply this transformation.
  virtual int ExpectedSequenceLength() const = 0;

  // Applies transformations to a sequence of nodes. Transformation
  // implementation is free manipulate with sequence nodes including adding
  // and/or deleting nodes. if there were updates to nodes in the end and/or
  // beginning of the sequence, then referential consistency should be
  // maintained by updating relevant references in nodes that precede this
  // sequence or depend on a last node of the sequence.
  virtual TransformResult ApplyToNodesSequence(
      const std::vector<Node*>& sequence, GraphFloat32* graph) = 0;
};

// A class accumulated decisions or updates done by transformations.
class TransformationReporter {
 public:
  virtual ~TransformationReporter() = default;

  virtual void DeclinedTransformation(const std::string& transformation,
                                      const std::string& node_ids,
                                      const std::string& message) = 0;

  virtual void AppliedTransformation(const std::string& transformation,
                                     const std::string& node_ids,
                                     const std::string& message) = 0;
};

// A class is designed to perform model transformations.
class ModelTransformer {
 public:
  ModelTransformer(GraphFloat32* graph, TransformationReporter* reporter)
      : graph_(graph), reporter_(reporter) {}

  // @return false if a graph is in the broken states can not be used any more
  bool Apply(const std::string& name, SequenceTransformation* transformation);

  // @return false if a graph is in the broken states can not be used any more
  bool Apply(const std::string& name, NodeTransformation* transformation);

 private:
  bool ApplyStartingWithNode(const std::string& name,
                             SequenceTransformation* transformation,
                             Node* begin);

  void AddNodeToProcess(Node* node) {
    if (node && processed_.insert(node->id).second) {
      to_process_.push_back(node->id);
    }
  }

  GraphFloat32* graph_;
  TransformationReporter* reporter_;

  std::deque<NodeId> to_process_;
  absl::flat_hash_set<NodeId> processed_;
};

class NullTransformationReporter : public TransformationReporter {
 public:
  void DeclinedTransformation(const std::string& transformation,
                              const std::string& nodes_id,
                              const std::string& message) override {}

  void AppliedTransformation(const std::string& transformation,
                             const std::string& nodes_id,
                             const std::string& message) override {}
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_MODEL_TRANSFORMER_H_
