/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BARRIER_REQUESTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_BARRIER_REQUESTS_H_

namespace xla::gpu {

// Collective thunks (including FFI calls) can request runtime to perform
// a barrier synchronization. Right now only global execution barrier is
// supported. This barrier makes sure that all devices have reached the same
// point of execution (both on GPU and CPU) which is needed to make sure that
// shared resources (such as multimem handlers) can be safely released.
class BarrierRequests {
 public:
  // TODO(484264395): Request barrier only for a specific clique.
  void RequestBarrierAfterModuleExecution() { need_barrier_ = true; }

  bool IsBarrierAfterModuleExecutionRequested() const { return need_barrier_; }

 private:
  bool need_barrier_ = false;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BARRIER_REQUESTS_H_
