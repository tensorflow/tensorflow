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
#include "tensorflow/core/tfrt/runtime/step_id.h"

#include <atomic>
#include <cstdint>

namespace tensorflow {
namespace tfrt_stub {

std::atomic<uint64_t>& GetGlobalInitialStepId() {
  static std::atomic<uint64_t> global_step_id = 0;
  return global_step_id;
}

TEST_ScopedInitialStepId::TEST_ScopedInitialStepId(uint64_t step_id) {
  step_id_ = GetGlobalInitialStepId().exchange(step_id);
}

TEST_ScopedInitialStepId::~TEST_ScopedInitialStepId() {
  GetGlobalInitialStepId().store(step_id_);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
