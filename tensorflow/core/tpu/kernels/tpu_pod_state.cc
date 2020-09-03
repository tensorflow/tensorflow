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
#include "tensorflow/core/tpu/kernels/tpu_pod_state.h"

#include "tensorflow/core/tpu/kernels/tpu_util.h"

namespace tensorflow {

const char kTpuPodStateResourceName[] = "tpu_pod_state";

TpuPodState::TpuPodState(
    int service_port, std::unique_ptr<TpuCompilationCacheService> cache_service)
    : cache_service_(std::move(cache_service)), service_port_(service_port) {}

TpuPodState::~TpuPodState() {
  if (cache_service_) {
    VLOG(1) << "Shutting down Compilation Cache Service.";
    if (cache_service_->Shutdown(20)) {
      if (service_port_ >= 0) {
        tpu::RecycleUnusedPort(service_port_);
      }
    } else {
      LOG(ERROR)
          << "Failed to shutdown Compilation Cache Service within timeout.";
    }
  }
  VLOG(1) << "Shutting down Compilation Cache Service done.";
}

string TpuPodState::DebugString() const {
  return "Wrapper for distributed TPU state";
}

Status GetTPUPodState(const ResourceMgr* rmgr, TpuPodState** pod_state) {
  if (!rmgr) {
    return errors::Internal("No resource manager.");
  }
  if (!rmgr->Lookup(rmgr->default_container(), kTpuPodStateResourceName,
                    pod_state)
           .ok()) {
    return errors::FailedPrecondition(
        "The TPU system has not been initialized.");
  }
  return Status::OK();
}

bool HasTPUPodState(const ResourceMgr* rmgr) {
  TpuPodState* pod_state;
  if (!rmgr->Lookup(rmgr->default_container(), kTpuPodStateResourceName,
                    &pod_state)
           .ok()) {
    return false;
  }
  pod_state->Unref();
  return true;
}

}  // namespace tensorflow
