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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PAD_FOR_TENSOR_CORES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PAD_FOR_TENSOR_CORES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Ensures that f16 cudnn convolutions have input/output channel dimensions that
// are multiples of 8, inserting pads/slices as necessary.
//
// This is useful primarily for Volta and newer GPUs, where tensor cores can
// only be used if the channel dims are multiples of 8.  It's probably the
// opposite of useful on other GPUs, so you should check what GPU you're
// targeting before running this pass.
//
// TODO(jlebar): Also pad dots.
class PadForTensorCores : public HloModulePass {
 public:
  absl::string_view name() const override { return "pad for tensor cores"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_PAD_FOR_TENSOR_CORES_H_
