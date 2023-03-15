/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_REGISTRY_H_

#include <functional>
#include <string>
#include <unordered_map>

#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/platform/thread_annotations.h"

namespace tensorflow {

typedef std::function<tensorflow::ImmediateExecutionContext*(
    const TFE_ContextOptions*)>
    ContextCreator;

struct ContextRegistry {
  mutex mu;
  std::unordered_map<string, ContextCreator> registry TF_GUARDED_BY(mu);

  StatusOr<ContextCreator> Lookup(string name) {
    tf_shared_lock l(mu);
    auto creator = registry.find(name);
    if (creator == registry.end()) {
      std::string error_msg =
          "TensorFlow EagerContext: " + name +
          " not found. Did you link the context registration library?";
      LOG(ERROR) << error_msg;
      return tsl::errors::NotFound(error_msg);
    } else {
      return creator->second;
    }
  }

  InitOnStartupMarker Register(string name, ContextCreator creator) {
    mutex_lock l(mu);
    CHECK(registry.find(name) == registry.end())  // Crash OK
        << "Eager context creator has already been registered: " << name;
    registry[name] = creator;
    return {};
  }
};

ContextRegistry* GlobalContextRegistry();

#define REGISTER_EAGER_CONTEXT_CREATOR(ctx, name, func)         \
  static ::tensorflow::InitOnStartupMarker const                \
      register_eager_context_creator##ctx TF_ATTRIBUTE_UNUSED = \
          TF_INIT_ON_STARTUP_IF(true)                           \
          << tensorflow::GlobalContextRegistry()->Register(name, func)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CONTEXT_REGISTRY_H_
