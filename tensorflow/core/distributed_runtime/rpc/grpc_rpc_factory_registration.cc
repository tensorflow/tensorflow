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

#include "tensorflow/core/distributed_runtime/rpc/grpc_rpc_factory.h"
#include "tensorflow/core/util/rpc/rpc_factory.h"
#include "tensorflow/core/util/rpc/rpc_factory_registry.h"

namespace tensorflow {
namespace {

// Used for adding the grpc factory to the RPC factory registry.
struct Value {
  static RPCFactory* Function(OpKernelConstruction* ctx, bool fail_fast,
                              int64 timeout_in_ms) {
    return new GrpcRPCFactory(ctx, fail_fast, timeout_in_ms);
  }
};

REGISTER_RPC_FACTORY("grpc", Value::Function);

}  // namespace
}  // namespace tensorflow
