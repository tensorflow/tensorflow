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

#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"

#include <deque>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"

namespace tflite {
namespace gpu {

bool ModelTransformer::Apply(const std::string& name,
                             SequenceTransformation* transformation) {
  // Seed transformations with starting node. Each node may start a chain of
  // transformations.
  for (auto input : graph_->inputs()) {
    for (auto node : graph_->FindConsumers(input->id)) {
      AddNodeToProcess(node);
    }
  }
  while (!to_process_.empty()) {
    auto node = graph_->GetNode(to_process_.front());
    if (node) {
      if (!ApplyStartingWithNode(name, transformation, node)) {
        return false;
      }
    }
    to_process_.pop_front();
  }
  processed_.clear();
  return true;
}

bool ModelTransformer::Apply(const std::string& name,
                             NodeTransformation* transformation) {
  // Apply a transformation only to nodes that are present in the graph before
  // transformation.
  std::vector<NodeId> nodes;
  for (auto node : graph_->nodes()) {
    nodes.push_back(node->id);
  }
  for (auto node_id : nodes) {
    auto node = graph_->GetNode(node_id);
    if (!node) {
      continue;
    }
    auto result = transformation->ApplyToNode(node, graph_);
    last_transformation_message_ = result.message;
    if (result.status == TransformStatus::INVALID) {
      return false;
    }
  }
  return true;
}

const std::string& ModelTransformer::last_transformation_message() const {
  return last_transformation_message_;
}

bool ModelTransformer::ApplyStartingWithNode(
    const std::string& name, SequenceTransformation* transformation,
    Node* begin) {
  int expected_sequence_length = transformation->ExpectedSequenceLength();

  std::deque<NodeId> sequence;
  std::vector<Node*> nodes;
  nodes.reserve(transformation->ExpectedSequenceLength());
  sequence.push_back(begin->id);

  // Go over nodes with sequence sliding window of size
  // expected_sequence_length until a node with multiple dependents is found.
  while (true) {
    // Apply transformation if possible.
    if (sequence.size() == expected_sequence_length) {
      nodes.clear();
      for (NodeId id : sequence) {
        // Nodes present in sequence should be present in a graph. If they are
        // not, then this transformation changes a graph but didn't say it.
        Node* node = graph_->GetNode(id);
        if (node == nullptr) {
          return false;
        }
        nodes.push_back(node);
      }

      NodeId first_in_sequence = sequence.front();
      auto preceding_node =
          graph_->FindProducer(graph_->FindInputs(first_in_sequence)[0]->id);
      auto result = transformation->ApplyToNodesSequence(nodes, graph_);
      last_transformation_message_ = result.message;
      if (result.status == TransformStatus::INVALID) {
        // graph is broken now.
        return false;
      }
      if (result.status == TransformStatus::APPLIED) {
        // Also remove first node of a sequence from a set of processed node.
        // Out of all nodes in a sequence only first one may have been added
        // to "processed" set because other nodes do not have more than one
        // dependent. However, if a sequence is changed, then processing needs
        // to be restarted again.
        processed_.erase(first_in_sequence);
        // Transformation was successful. Restart sequence from the node that
        // precedes current sequence.
        if (preceding_node) {
          processed_.erase(preceding_node->id);
          AddNodeToProcess(preceding_node);
        } else {
          // This is the first node in the graph. Re-seed transformation.
          for (auto input : graph_->inputs()) {
            for (auto node : graph_->FindConsumers(input->id)) {
              AddNodeToProcess(node);
            }
          }
        }
        return true;
      }
    }

    // Try to extend current sequence.
    Node* next_node_in_sequence = nullptr;
    bool has_multiple_children = false;

    // Check that all outputs from last node are consumed by a single node.
    for (auto output_value : graph_->FindOutputs(sequence.back())) {
      for (auto dependent : graph_->FindConsumers(output_value->id)) {
        if (has_multiple_children) {
          AddNodeToProcess(dependent);
        } else if (next_node_in_sequence == nullptr) {
          next_node_in_sequence = dependent;
        } else if (next_node_in_sequence != dependent) {
          // There are more than two nodes depend on the output from end node,
          // therefore here a sequence stops and new will start. Push all such
          // nodes.
          has_multiple_children = true;
          AddNodeToProcess(dependent);
          AddNodeToProcess(next_node_in_sequence);
        }
      }
    }

    // Now check that next node has inputs only produced by the last node.
    if (!has_multiple_children && next_node_in_sequence) {
      for (auto input : graph_->FindInputs(next_node_in_sequence->id)) {
        auto producer = graph_->FindProducer(input->id);
        if (producer == nullptr || producer->id != sequence.back()) {
          has_multiple_children = true;
          AddNodeToProcess(next_node_in_sequence);
          break;
        }
      }
    }

    if (has_multiple_children || next_node_in_sequence == nullptr) {
      // reached end of this transformation sequence.
      return true;
    }

    sequence.push_back(next_node_in_sequence->id);
    // Decrease sequence until it matches expected length.
    if (sequence.size() > expected_sequence_length) {
      sequence.pop_front();
    }
  }
  return true;
}

}  // namespace gpu
}  // namespace tflite
