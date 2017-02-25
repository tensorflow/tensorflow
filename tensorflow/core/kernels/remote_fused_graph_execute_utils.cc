/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"

#include <utility>

namespace tensorflow {

RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar::ExecutorBuildRegistrar(
    const string& name, ExecutorBuildFunc executor_build_func) {
  ExecutorBuildRegistry& executor_build_registry = *GetExecutorBuildRegistry();
  executor_build_registry[name] = std::move(executor_build_func);
}

/* static */ const RemoteFusedGraphExecuteUtils::ExecutorBuildFunc*
RemoteFusedGraphExecuteUtils::GetExecutorBuildFunc(const string& name) {
  ExecutorBuildRegistry& executor_build_registry = *GetExecutorBuildRegistry();
  if (executor_build_registry.count(name) <= 0) {
    return nullptr;
  }
  return &executor_build_registry.at(name);
}

/* static */ RemoteFusedGraphExecuteUtils::ExecutorBuildRegistry*
RemoteFusedGraphExecuteUtils::GetExecutorBuildRegistry() {
  static ExecutorBuildRegistry executor_builder_registry;
  return &executor_builder_registry;
}

}  // namespace tensorflow
