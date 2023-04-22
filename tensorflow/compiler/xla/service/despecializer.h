/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DESPECIALIZER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DESPECIALIZER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Creates an HloPassPipeline containing multiple HloPasses that can
// despecialize an optimized HloModule. This is useful to run an HloModule
// optimized for one specific platform on a different platform (undoing platform
// specific passes) with matching numerics for comparison.
//
// Current despecialization passes are HloDescheduler, ControlDepRemover,
// Defuser and BFloat16MixedPrecisionRemoval.
class Despecializer : public HloModulePass {
 public:
  Despecializer();
  absl::string_view name() const override { return "despecializer"; }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  HloPassPipeline pipeline_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DESPECIALIZER_H_
