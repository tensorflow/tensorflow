/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/xla_config_registry.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace xla_config_registry {

namespace {
struct GlobalJitLevelState {
  mutex mu;
  GlobalJitLevelGetterTy getter GUARDED_BY(mu);
};

GlobalJitLevelState* GetSingletonState() {
  static GlobalJitLevelState* state = new GlobalJitLevelState;
  return state;
}
}  // namespace

void RegisterGlobalJitLevelGetter(GlobalJitLevelGetterTy getter) {
  GlobalJitLevelState* state = GetSingletonState();
  mutex_lock l(state->mu);
  CHECK(!state->getter);
  state->getter = std::move(getter);
}

XlaGlobalJitLevel GetGlobalJitLevel(
    OptimizerOptions::GlobalJitLevel jit_level_in_session_opts) {
  GlobalJitLevelState* state = GetSingletonState();
  mutex_lock l(state->mu);
  if (!state->getter) {
    return {jit_level_in_session_opts, jit_level_in_session_opts};
  }
  return state->getter(jit_level_in_session_opts);
}

}  // namespace xla_config_registry

}  // namespace tensorflow
