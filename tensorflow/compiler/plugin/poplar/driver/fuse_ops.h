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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_FUSE_OPS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_FUSE_OPS_H_

#include "tensorflow/compiler/xla/service/instruction_fusion.h"

namespace xla {

class HloModule;

namespace poplarplugin {

class FuseOps : public InstructionFusion {
public:
  FuseOps() : InstructionFusion(InstructionFusion::IsExpensive, true) {}

  ~FuseOps() override = default;

  tensorflow::StringPiece name() const override { return "poplar-fuse"; }

protected:
  bool ShouldFuse(HloInstruction*, int64) override;

  HloInstruction::FusionKind ChooseKind(const HloInstruction*,
                                        const HloInstruction*) override;
};

}
}

#endif
