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
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb.h"

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/acceleration/compatibility/database_generated.h"

namespace tflite {
namespace acceleration {
namespace {

std::vector<const DeviceDecisionTreeEdge*> Find(
    const DeviceDecisionTreeNode* root, const std::string& value) {
  std::vector<const DeviceDecisionTreeEdge*> found;
  if (root->comparison() == Comparison_EQUAL) {
    // Exact match.
    const DeviceDecisionTreeEdge* possible =
        root->items()->LookupByKey(value.c_str());
    if (possible) {
      found.push_back(possible);
    }
  } else {
    // Minimum/Maximum: value should be at least / at most item's value.
    for (const DeviceDecisionTreeEdge* item : *(root->items())) {
      if ((root->comparison() == Comparison_MINIMUM)
              ? value >= item->value()->str()
              : value <= item->value()->str()) {
        found.push_back(item);
      }
    }
  }
  return found;
}

void UpdateVariablesFromDeviceDecisionTreeEdges(
    std::map<std::string, std::string>* variable_values,
    const DeviceDecisionTreeEdge& item) {
  if (item.derived_properties()) {
    for (const DerivedProperty* p : *(item.derived_properties())) {
      (*variable_values)[p->variable()->str()] = p->value()->str();
    }
  }
}

void Follow(const DeviceDecisionTreeNode* root,
            std::map<std::string, std::string>* variable_values) {
  if (!root->variable()) {
    return;
  }
  auto possible_value = variable_values->find(root->variable()->str());
  if (possible_value == variable_values->end()) {
    return;
  }
  std::vector<const DeviceDecisionTreeEdge*> edges =
      Find(root, possible_value->second);
  for (const DeviceDecisionTreeEdge* edge : edges) {
    UpdateVariablesFromDeviceDecisionTreeEdges(variable_values, *edge);
    if (edge->children()) {
      for (const DeviceDecisionTreeNode* root : *(edge->children())) {
        Follow(root, variable_values);
      }
    }
  }
}

}  // namespace

void UpdateVariablesFromDatabase(
    std::map<std::string, std::string>* variable_values,
    const DeviceDatabase& database) {
  if (!database.root()) return;
  for (const DeviceDecisionTreeNode* root : *(database.root())) {
    Follow(root, variable_values);
  }
}

}  // namespace acceleration
}  // namespace tflite
