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

#include "tensorflow/core/util/rpc/rpc_factory.h"

#include "tensorflow/core/util/rpc/rpc_factory_registry.h"

namespace tensorflow {

RPCFactoryRegistry* RPCFactoryRegistry::Global() {
  static RPCFactoryRegistry* registry = new RPCFactoryRegistry;
  return registry;
}

RPCFactoryRegistry::RPCFactoryFn* RPCFactoryRegistry::Get(
    const string& protocol) {
  auto found = fns_.find(protocol);
  if (found == fns_.end()) return nullptr;
  return &found->second;
}

void RPCFactoryRegistry::Register(const string& protocol,
                                  const RPCFactoryFn& factory_fn) {
  auto existing = Get(protocol);
  CHECK_EQ(existing, nullptr)
      << "RPC factory for protocol: " << protocol << " already registered";
  fns_.insert(std::pair<const string&, RPCFactoryFn>(protocol, factory_fn));
}

}  // namespace tensorflow
