/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/executor_factory.h"

#include <unordered_map>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

mutex& executor_factory_lock() {
  static mutex mu(LINKER_INITIALIZED);
  return mu;
}

typedef std::unordered_map<string, ExecutorFactory*> ExecutorFactories;
ExecutorFactories* executor_factories() {
  static ExecutorFactories* factories = new ExecutorFactories;
  return factories;
}

}  // namespace

void ExecutorFactory::Register(const string& executor_type,
                               ExecutorFactory* factory) {
  mutex_lock l(executor_factory_lock());
  if (!executor_factories()->insert({executor_type, factory}).second) {
    LOG(FATAL) << "Two executor factories are being registered "
               << "under" << executor_type;
  }
}

namespace {
const string RegisteredFactoriesErrorMessageLocked()
    TF_SHARED_LOCKS_REQUIRED(executor_factory_lock()) {
  std::vector<string> factory_types;
  for (const auto& executor_factory : *executor_factories()) {
    factory_types.push_back(executor_factory.first);
  }
  return strings::StrCat("Registered factories are {",
                         absl::StrJoin(factory_types, ", "), "}.");
}
}  // namespace

Status ExecutorFactory::GetFactory(const string& executor_type,
                                   ExecutorFactory** out_factory) {
  tf_shared_lock l(executor_factory_lock());

  auto iter = executor_factories()->find(executor_type);
  if (iter == executor_factories()->end()) {
    return errors::NotFound(
        "No executor factory registered for the given executor type: ",
        executor_type, " ", RegisteredFactoriesErrorMessageLocked());
  }

  *out_factory = iter->second;
  return OkStatus();
}

Status NewExecutor(const string& executor_type,
                   const LocalExecutorParams& params, const Graph& graph,
                   std::unique_ptr<Executor>* out_executor) {
  ExecutorFactory* factory = nullptr;
  TF_RETURN_IF_ERROR(ExecutorFactory::GetFactory(executor_type, &factory));
  return factory->NewExecutor(params, std::move(graph), out_executor);
}

}  // namespace tensorflow
