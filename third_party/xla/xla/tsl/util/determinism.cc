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

#include "xla/tsl/util/determinism.h"

#include "absl/strings/string_view.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/mutex.h"

namespace tsl {

namespace {

class DeterminismState {
 public:
  explicit DeterminismState(absl::string_view env_var) : env_var_(env_var) {}
  bool Required() {
    mutex_lock l(*mutex_);

    if (state_ == Value::NOT_SET) {
      bool env_var_set = false;
      TF_CHECK_OK(tsl::ReadBoolFromEnvVar(env_var_,
                                          /*default_val=*/false, &env_var_set));
      state_ = env_var_set ? Value::ENABLED : Value::DISABLED;
    }

    return state_ == Value::ENABLED;
  }
  void Enable(bool enabled) {
    mutex_lock l(*mutex_);
    state_ = enabled ? Value::ENABLED : Value::DISABLED;
  }

 private:
  absl::string_view env_var_;
  enum class Value { DISABLED, ENABLED, NOT_SET };
  mutex* mutex_ = new mutex;
  Value state_ = Value::NOT_SET;
};

}  // namespace

DeterminismState OpDeterminismState = DeterminismState("TF_DETERMINISTIC_OPS");
DeterminismState OpOrderDeterminismState =
    DeterminismState("TF_DETERMINISTIC_ORDER");

bool OpDeterminismRequired() { return OpDeterminismState.Required(); }
void EnableOpDeterminism(bool enabled) { OpDeterminismState.Enable(enabled); }
bool OpOrderDeterminismRequired() { return OpOrderDeterminismState.Required(); }

}  // namespace tsl
