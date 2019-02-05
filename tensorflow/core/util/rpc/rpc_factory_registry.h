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

#ifndef TENSORFLOW_CORE_UTIL_RPC_RPC_FACTORY_REGISTRY_H_
#define TENSORFLOW_CORE_UTIL_RPC_RPC_FACTORY_REGISTRY_H_

#include <map>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/rpc/rpc_factory.h"

namespace tensorflow {

class RPCFactoryRegistry {
 public:
  typedef std::function<RPCFactory*(OpKernelConstruction* ctx, bool fail_fast,
                                    int64 timeout_in_ms)>
      RPCFactoryFn;

  // Returns a pointer to a global RPCFactoryRegistry object.
  static RPCFactoryRegistry* Global();

  // Returns a pointer to an function that creates an RPC factory for the given
  // protocol.
  RPCFactoryFn* Get(const string& protocol);

  // Registers a function that creates and RPC factory for the given protocol.
  // The function should transfer the ownership of the factory to its caller.
  void Register(const string& protocol, const RPCFactoryFn& factory_fn);

 private:
  std::map<string, RPCFactoryFn> fns_;
};

namespace rpc_factory_registration {

class RPCFactoryRegistration {
 public:
  RPCFactoryRegistration(const string& protocol,
                         const RPCFactoryRegistry::RPCFactoryFn& factory_fn) {
    RPCFactoryRegistry::Global()->Register(protocol, factory_fn);
  }
};

}  // namespace rpc_factory_registration

#define REGISTER_RPC_FACTORY(protocol, factory_fn) \
  REGISTER_RPC_FACTORY_UNIQ_HELPER(__COUNTER__, protocol, factory_fn)

#define REGISTER_RPC_FACTORY_UNIQ_HELPER(ctr, protocol, factory_fn) \
  REGISTER_RPC_FACTORY_UNIQ(ctr, protocol, factory_fn)

#define REGISTER_RPC_FACTORY_UNIQ(ctr, protocol, factory_fn) \
  static rpc_factory_registration::RPCFactoryRegistration    \
      rpc_factory_registration_fn_##ctr(protocol, factory_fn)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_RPC_RPC_FACTORY_REGISTRY_H_
