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

#include "tensorflow/compiler/xla/service/hlo_phi_graph.h"

#include <queue>

namespace xla {
HloValue::Id PhiGraph::GetOptimizedId(const HloValue& value) {
  Node* node = value_id_to_node_[value.id()];
  return node->value_id;
}

// Returns true if the input to a hlo value is the same as `inputs`.
bool PhiGraph::InputsEqualTo(const HloValue& value,
                             absl::Span<const HloValue* const> inputs) {
  auto iter = value_id_to_node_.find(value.id());
  CHECK(iter != value_id_to_node_.end());
  absl::flat_hash_set<HloValue::Id> existing_set;
  for (Node* operand : iter->second->operands) {
    existing_set.insert(operand->value_id);
  }
  absl::flat_hash_set<HloValue::Id> new_set;
  for (const HloValue* input : inputs) {
    new_set.insert(input->id());
  }
  return existing_set == new_set;
}

HloValue::Id PhiGraph::FindOptimizedValue(const HloValue::Id id) {
  auto iter = value_id_to_node_.find(id);
  CHECK(iter != value_id_to_node_.end());
  return iter->second->value_id;
}

PhiGraph::Node* PhiGraph::CreateOrReuseNode(const HloValue& value) {
  auto iter = value_id_to_node_.find(value.id());
  if (iter == value_id_to_node_.end()) {
    node_storage_.emplace_back(absl::make_unique<Node>());
    Node* node = node_storage_.back().get();
    node->value_id = value.id();
    value_id_to_node_[value.id()] = node;
    node_to_value_id_[node].push_back(value.id());
    return node;
  } else {
    // A node is already registered with this value, check the value_id
    // is the same as previously registrated.
    CHECK_NE(iter->second, nullptr);
    CHECK_EQ(iter->second->value_id, value.id());
    return iter->second;
  }
}

void PhiGraph::ReplaceNodeWith(PhiGraph::Node* node, PhiGraph::Node* replace) {
  // Update users.
  CHECK(node->is_phi);
  for (Node* user : node->users) {
    absl::c_replace(user->operands, node, replace);
  }

  // Update operand's users
  for (Node* operand : node->operands) {
    absl::c_replace(operand->users, node, replace);
  }
  for (HloValue::Id value_id : node_to_value_id_[node]) {
    CHECK(value_id_to_node_.contains(value_id));
    value_id_to_node_[value_id] = replace;
  }
  // Update mappings to HloValue::Id.
  absl::c_copy(node_to_value_id_[node],
               std::back_inserter(node_to_value_id_[replace]));
  node_to_value_id_[node].clear();
  node->mark_as_dead = true;
}

void PhiGraph::RegisterPhi(const HloValue& value,
                           absl::Span<const HloValue* const> inputs) {
  Node* node = CreateOrReuseNode(value);
  CHECK(value.is_phi());
  node->is_phi = true;
  node->operands.clear();
  for (auto input : inputs) {
    CHECK(input != nullptr);
    Node* input_node = CreateOrReuseNode(*input);
    node->operands.push_back(input_node);
  }
}

std::string PhiGraph::ToString() {
  std::string out = "PhiGraph: \n";
  for (auto& node : node_storage_) {
    std::string is_phi = node->is_phi ? ", phi" : "";
    std::string is_optimized = node->mark_as_dead ? ", dead" : "";
    absl::StrAppend(&out, node->value_id);
    absl::StrAppend(&out, is_phi);
    absl::StrAppend(&out, is_optimized, ":\n");
    for (Node* input : node->operands) {
      absl::StrAppend(&out, "  ", input->value_id);
      absl::StrAppend(&out, "\n");
    }
  }
  return out;
}

void PhiGraph::Optimize() {
  // Set up users for each node.
  for (auto& node : node_storage_) {
    for (Node* input : node->operands) {
      input->users.push_back(node.get());
    }
  }

  // input_node->users.push_back(node);
  bool changed = true;

  // Run the optimization to a fixed point.
  while (changed) {
    changed = false;
    absl::flat_hash_set<Node*> checked_for_closure;
    for (auto& node : node_storage_) {
      // Only optimize phi node.
      if (!node->is_phi) {
        continue;
      }
      // Skip dead nodes
      if (node->mark_as_dead) {
        continue;
      }

      Node* node_ptr = node.get();

      CHECK_GE(node_ptr->operands.size(), 1);

      // Remove self-referencing ids from users and operands.
      auto it = absl::c_find(node_ptr->operands, node_ptr);
      while (it != node_ptr->operands.end()) {
        node_ptr->operands.erase(it);
        it = absl::c_find(node_ptr->operands, node_ptr);
      }

      it = absl::c_find(node_ptr->users, node_ptr);
      while (it != node_ptr->users.end()) {
        node_ptr->users.erase(it);
        it = absl::c_find(node_ptr->users, node_ptr);
      }

      // If all inputs to phi (after self referencing ids are removed) are the
      // same value, replace the phi with that value.
      //
      // phi(A, A, ... A) => A
      // phi(A, self) = phi(A) => A
      CHECK_GE(node_ptr->operands.size(), 1);
      bool all_inputs_are_same = absl::c_all_of(
          node_ptr->operands,
          [&](Node* elem) { return elem == node_ptr->operands[0]; });

      if (all_inputs_are_same) {
        ReplaceNodeWith(node_ptr, node_ptr->operands[0]);
        changed = true;
        continue;
      }

      // Find a closure of inter-connected phis and one non-phi node. Replace
      // all phis with that non-phi node.
      //
      // def A = phi(B, C)
      // def B = phi(C, D)
      // def C = phi(A, B)
      // def D = non-phi
      // Replace A, B, and C with D:
      // A = phi(B, C) => D
      // B = phi(C, D) => D
      // C = phi(A, B) => D
      if (checked_for_closure.contains(node_ptr)) {
        continue;
      }
      // Keeps track of nodes in the current closure being tested.
      absl::flat_hash_set<Node*> workset;
      std::queue<Node*> worklist;
      Node* non_phi = nullptr;
      worklist.push(node_ptr);
      while (!worklist.empty()) {
        Node* todo = worklist.front();
        worklist.pop();
        if (workset.contains(todo)) {
          continue;
        }
        checked_for_closure.insert(todo);
        workset.insert(todo);
        for (Node* operand : todo->operands) {
          worklist.push(operand);
        }
        if (!todo->is_phi) {
          if (non_phi != nullptr && non_phi != todo) {
            // We see distinct non-phi nodes in the closure, can't apply the
            // optimization.
            non_phi = nullptr;
            // Break the while loop non_phi setting to nullptr, signaling that
            // the optimization can't be applied.
            break;
          } else {
            // This is the non_phi node we are seeing so far.
            non_phi = todo;
          }
        }
      }
      if (non_phi != nullptr) {
        // Replace all phi nodes in the closure/workset with the non_phi node.
        for (Node* node : workset) {
          if (!node->is_phi) {
            CHECK_EQ(node, non_phi);
            continue;
          }
          ReplaceNodeWith(node, non_phi);
          changed = true;
        }
      }
    }
  }
}
}  // namespace xla
