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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_POD_STATE_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_POD_STATE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_service.h"

namespace tensorflow {

// Name of tpu pod state.
ABSL_CONST_INIT extern const char kTpuPodStateResourceName[];

// Wrapper to hold centralized state for the distributed TPU in the TPU_SYSTEM
// device's resource manager.
class TpuPodState : public ResourceBase {
 public:
  // The port number given by isa_cache_port will be freed with
  // RecycleUnusedPort in the destructor if it is non-negative.
  TpuPodState(int service_port,
              std::unique_ptr<TpuCompilationCacheService> cache_service);

  ~TpuPodState() override;

  string DebugString() const override;

 private:
  std::unique_ptr<TpuCompilationCacheService> cache_service_;
  int service_port_;
};

// Returns the TPU pod state or an error.
Status GetTPUPodState(const ResourceMgr* rmgr, TpuPodState** pod_state);

// Checks whether the TPU POD state configuration is present within the resource
// manager.
bool HasTPUPodState(const ResourceMgr* rmgr);

// Construct TpuPodState.
Status ConstructTpuPodState(
    ResourceMgr* rmgr, const std::vector<int32_t>& num_devices_per_host,
    tpu::TpuCompilationCacheInterface* compilation_cache,
    std::string* host_config_proto);

Status GetServerAddressAndPort(std::string* server_address, int* serving_port);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_POD_STATE_H_
