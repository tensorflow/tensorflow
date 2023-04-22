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

#include "tensorflow/core/platform/enable_tf2_utils.h"

#include <atomic>

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

enum Enablement : uint8 { kFalse = 0, kTrue = 1, undefined = 2 };

// If this flag is set, we will use it as a signal to decide on whether to
// use the MLIR based TF-XLA bridge.
static std::atomic<Enablement> tf2_enabled{undefined};

// Determine whether or not the user has explicitly asked for tf2 execution.
// Will be used to determine whether to use the MLIR based bridge.
void set_tf2_execution(bool enabled) {
  tf2_enabled = (enabled) ? Enablement::kTrue : Enablement::kFalse;
}

bool tf2_execution_enabled() {
  if (tf2_enabled == Enablement::undefined) {
    static bool tf2_behavior_env_enabled = [] {
      string tf2_env;
      TF_CHECK_OK(ReadStringFromEnvVar("TF2_BEHAVIOR", "0", &tf2_env));
      return tf2_env != "0";
    }();
    tf2_enabled =
        (tf2_behavior_env_enabled) ? Enablement::kTrue : Enablement::kFalse;
  }
  return tf2_enabled;
}

}  // namespace tensorflow
