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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_WHITELIST_DECISION_TREE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_WHITELIST_DECISION_TREE_H_

#include <map>
#include <string>

#include "tensorflow/lite/experimental/acceleration/whitelist/database_generated.h"

namespace tflite {
namespace acceleration {

// Use the variables in `variable_values` to evaluate the decision tree in
// `database` and update the `variable_values` based on derived properties in
// the decision tree.
//
// See database.fbs for a description of the decision tree.
void UpdateVariablesFromDatabase(
    std::map<std::string, std::string>* variable_values,
    const DeviceDatabase& database);

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_WHITELIST_DECISION_TREE_H_
