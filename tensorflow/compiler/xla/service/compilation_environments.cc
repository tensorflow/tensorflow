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

#include "tensorflow/compiler/xla/service/compilation_environments.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

CompilationEnvironments& CompilationEnvironments::operator=(
    const CompilationEnvironments& rhs) {
  Clear();
  for (const auto& descriptor_message_pair : rhs.environments_) {
    auto env = absl::WrapUnique(descriptor_message_pair.second->New());
    env->CopyFrom(*descriptor_message_pair.second);
    environments_.insert({descriptor_message_pair.first, std::move(env)});
  }
  return *this;
}

void CompilationEnvironments::AddEnv(
    std::unique_ptr<tensorflow::protobuf::Message> env) {
  auto descriptor = env->GetDescriptor();
  if (environments_.contains(descriptor)) {
    LOG(WARNING) << "Replacing CompilationEnvironment of type "
                 << descriptor->full_name();
  }

  environments_.insert({descriptor, std::move(env)});
}

}  // namespace xla
