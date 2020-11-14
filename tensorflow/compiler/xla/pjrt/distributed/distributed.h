/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_DISTRIBUTED_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_DISTRIBUTED_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// APIs for starting the distributed runtime service and client. Note that these
// variants use insecure credentials; the functions to build the service and
// client are kept separate so that other implementations using more secure
// credentials may be provided by the user.

// Builds a distributed runtime service. `address` is the address on which
// the service should listen, e.g., [::]:1234 . `num_nodes` is the number
// of nodes in the cluster.
StatusOr<std::unique_ptr<DistributedRuntimeService>>
GetDistributedRuntimeService(std::string address, int num_nodes);

// Builds a distributed runtime client, connecting to a service at `address`,
// where address is a gRPC-style address such as `dns:///localhost:1234`.
std::shared_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::string address);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_DISTRIBUTED_H_
