/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/determinism.h"

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

enum class DeterminismState { DISABLED, ENABLED, NOT_SET };
mutex* determinism_state_mutex = new mutex;
DeterminismState determinism_state = DeterminismState::NOT_SET;

}  // namespace

bool OpDeterminismRequired() {
  mutex_lock l(*determinism_state_mutex);

  if (determinism_state == DeterminismState::NOT_SET) {
    bool env_var_set = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                               /*default_val=*/false,
                                               &env_var_set));
    determinism_state =
        env_var_set ? DeterminismState::ENABLED : DeterminismState::DISABLED;
  }

  return determinism_state == DeterminismState::ENABLED;
}

void EnableOpDeterminism(bool enabled) {
  mutex_lock l(*determinism_state_mutex);
  determinism_state =
      enabled ? DeterminismState::ENABLED : DeterminismState::DISABLED;
}

}  // namespace tensorflow
