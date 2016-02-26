/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_

namespace tensorflow {

// Represents an abstract asynchronous service that handles incoming
// RPCs with a polling loop.
class AsyncServiceInterface {
 public:
  virtual ~AsyncServiceInterface() {}

  // A blocking method that should be called to handle incoming RPCs.
  // This method will block until the service is shutdown, which
  // depends on the implementation of the service.
  virtual void HandleRPCsLoop() = 0;

  // TODO(mrry): Add a clean shutdown method?
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_ASYNC_SERVICE_INTERFACE_H_
