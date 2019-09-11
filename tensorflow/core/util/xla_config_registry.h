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

#ifndef TENSORFLOW_CORE_UTIL_XLA_CONFIG_REGISTRY_H
#define TENSORFLOW_CORE_UTIL_XLA_CONFIG_REGISTRY_H

#include <functional>
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A registry class where XLA can register callbacks for Tensorflow to query
// its status.
class XlaConfigRegistry {
 public:
  // Input is jit_level in session config, and return is the config from
  // XLA, reflecting the effect of the environment variable flags.
  typedef std::function<OptimizerOptions::GlobalJitLevel(
      OptimizerOptions::GlobalJitLevel)>
      global_jit_level_getter_t;

  static bool Register(XlaConfigRegistry::global_jit_level_getter_t getter) {
    CHECK(!global_jit_level_getter_);
    global_jit_level_getter_ = std::move(getter);
    return true;
  }

  static OptimizerOptions::GlobalJitLevel GetGlobalJitLevel(
      OptimizerOptions::GlobalJitLevel jit_level_in_session_opts) {
    if (!global_jit_level_getter_) {
      return jit_level_in_session_opts;
    }
    return global_jit_level_getter_(jit_level_in_session_opts);
  }

 private:
  static global_jit_level_getter_t global_jit_level_getter_;
};

#define REGISTER_XLA_CONFIG_GETTER(getter) \
  static bool registered_##__COUNTER__ =   \
      ::tensorflow::XlaConfigRegistry::Register(getter)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_XLA_CONFIG_REGISTRY_H
