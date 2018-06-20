/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UPDATE_OP_DEPENDENCIES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_UPDATE_OP_DEPENDENCIES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

struct CompilerAnnotations;

// Add control dependencies between in-place updates and readers of the updates
// (must come after any outliners, eg ExpressionOutliner, FuseOps, Outliner)

class UpdateOpDependenctOrdering : public HloPassInterface {
 public:
  UpdateOpDependenctOrdering(CompilerAnnotations& annotations) :
      annotations_(annotations) {}

  ~UpdateOpDependenctOrdering() override = default;

  tensorflow::StringPiece name() const override {
    return "update-op-dependencies";
  }

  StatusOr<bool> Run(HloModule *module) override;

 private:
  CompilerAnnotations& annotations_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
