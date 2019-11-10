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

#include "tensorflow/core/grappler/optimizers/data/vectorization/vectorizer_registry.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

VectorizerRegistry* VectorizerRegistry::Global() {
  static VectorizerRegistry* registry = new VectorizerRegistry;
  return registry;
}

Vectorizer* VectorizerRegistry::Get(const string& op_type) {
  auto found = vectorizers_.find(op_type);
  if (found == vectorizers_.end()) {
    return nullptr;
  }
  return found->second.get();
}

void VectorizerRegistry::Register(const string& op_type,
                                  std::unique_ptr<Vectorizer> vectorizer) {
  auto existing = Get(op_type);
  CHECK_EQ(existing, nullptr)
      << "Vectorizer for op type: " << op_type << " already registered";
  vectorizers_.insert(std::pair<const string&, std::unique_ptr<Vectorizer>>(
      op_type, std::move(vectorizer)));
}
}  // namespace grappler
}  // namespace tensorflow
