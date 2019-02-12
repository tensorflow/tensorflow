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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_ROOT_TOKEN_TRANSFORMER_PASS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_ROOT_TOKEN_TRANSFORMER_PASS_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/**
 * If the root instruction outputs a token then an empty tuple is added
 * as a root instruction with the token output added as a control dependency.
 *
 */
class RootTokenReplacer : public HloModulePass {
 public:
  RootTokenReplacer();

  ~RootTokenReplacer() = default;

  absl::string_view name() const override { return "root-token-replacer-pass"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
