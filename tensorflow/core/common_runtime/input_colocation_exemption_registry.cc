/* Copyright 2019 The TensorFlow Authors. Al Rights Reserved.

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
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"

#include <set>
#include <string>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

InputColocationExemptionRegistry* InputColocationExemptionRegistry::Global() {
  static InputColocationExemptionRegistry* registry =
      new InputColocationExemptionRegistry;
  return registry;
}

void InputColocationExemptionRegistry::Register(const string& op) {
  auto it = ops_.find(op);
  if (it != ops_.end()) {
    LOG(WARNING) << "Input colocation exemption for op: " << op
                 << " already registered";
  } else {
    ops_.insert(op);
  }
}

}  // namespace tensorflow
