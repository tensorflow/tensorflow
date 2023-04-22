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

#include <string>

#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/util/proto/descriptor_pool_registry.h"

namespace tensorflow {

DescriptorPoolRegistry* DescriptorPoolRegistry::Global() {
  static DescriptorPoolRegistry* registry = new DescriptorPoolRegistry;
  return registry;
}

DescriptorPoolRegistry::DescriptorPoolFn* DescriptorPoolRegistry::Get(
    const string& source) {
  auto found = fns_.find(source);
  if (found == fns_.end()) return nullptr;
  return &found->second;
}

void DescriptorPoolRegistry::Register(
    const string& source,
    const DescriptorPoolRegistry::DescriptorPoolFn& pool_fn) {
  auto existing = Get(source);
  CHECK_EQ(existing, nullptr)
      << "descriptor pool for source: " << source << " already registered";
  fns_.insert(std::pair<const string&, DescriptorPoolFn>(source, pool_fn));
}

}  // namespace tensorflow
