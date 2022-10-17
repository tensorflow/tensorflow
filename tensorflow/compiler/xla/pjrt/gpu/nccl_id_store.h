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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_GPU_NCCL_ID_STORE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_GPU_NCCL_ID_STORE_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// A table mapping NcclCliqueKeys to ncclUniqueId values encoded as strings.
// In a distributed setup the table of NCCL IDs is kept on the master node
// (node 0). The node of the first participating device will create the unique
// id.
class NcclIdStore {
 public:
  NcclIdStore(int node_id, std::shared_ptr<DistributedRuntimeClient> client,
              absl::flat_hash_map<GlobalDeviceId, int> device_to_node)
      : node_id_(node_id),
        client_(std::move(client)),
        device_to_node_(std::move(device_to_node)) {}

  StatusOr<std::string> GetNcclUniqueId(const gpu::NcclCliqueKey& key);

 private:
  const int node_id_;
  const std::shared_ptr<DistributedRuntimeClient> client_;
  const absl::flat_hash_map<GlobalDeviceId, int> device_to_node_;

  absl::Mutex mu_;
  absl::flat_hash_map<gpu::NcclCliqueKey, std::string> cache_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_GPU_NCCL_ID_STORE_H_
