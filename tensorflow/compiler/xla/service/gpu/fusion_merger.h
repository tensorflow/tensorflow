/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_MERGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_MERGER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// An HLO pass that attempts to merge fusion instructions to reduce kernel
// launch overhead and improve data locality.
//
// Fusion instructions are merged into their users if two conditions are met:
//
// 1) The flops_to_bytes ratio of the fusion instruction is below the threshold
//    value of 1.0.
// 2) The result of merging the fusion instruction into its users would not
//    increase bytes transferred.
//
class FusionMerger : public HloPassInterface {
 public:
  tensorflow::StringPiece name() const override { return "fusion merger"; }

  StatusOr<bool> Run(HloModule* module) override;

  static double GetThresholdFlopsToBytesRatio() { return 1.0; }
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSION_MERGER_H_
