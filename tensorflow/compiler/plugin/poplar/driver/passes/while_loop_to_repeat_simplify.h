/* Copyright 2018 Graphcore Ltd

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_WHILE_LOOP_TO_REPEAT_SIMPLIFY_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_WHILE_LOOP_TO_REPEAT_SIMPLIFY_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class HloModule;

namespace poplarplugin {

/** This pass tries to find the number of times a while loop is executed. If a
 * while loop can be simplified to a repeat, then the while instruction is
 * inserted into the map along with the number of iterations */
class WhileLoopToRepeatSimplify : public HloModulePass {
 public:
  WhileLoopToRepeatSimplify();
  ~WhileLoopToRepeatSimplify() override = default;

  absl::string_view name() const override {
    return "while-loop-to-repeat-simplify";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_WHILE_LOOP_TO_REPEAT_SIMPLIFY_H_
