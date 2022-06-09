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

#include "tensorflow/core/platform/logging.h"

namespace xla {

void CompilationEnvironments::AddEnv(EnvWrapper env) {
  if (environments_.contains(env.EnvTypeid())) {
    LOG(WARNING) << "Replacing CompilationEnvironment of type "
                 << env.EnvTypeid().name();
  }
  auto id = env.EnvTypeid();
  environments_.insert({id, std::move(env)});
}

}  // namespace xla
