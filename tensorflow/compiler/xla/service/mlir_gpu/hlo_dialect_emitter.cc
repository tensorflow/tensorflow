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

#include "tensorflow/compiler/xla/service/mlir_gpu/hlo_dialect_emitter.h"

namespace xla {
namespace gpu {

HloDialectEmitter::HloDialectEmitter(const HloModule& hlo_module,
                                     const BufferAssignment& assignment,
                                     ::mlir::ModuleOp mlir_module)
    : mlir_module_(mlir_module), builder_(mlir_module_.getContext()) {}

Status DefaultAction(HloInstruction* hlo) {
  LOG(FATAL) << "Not implemented yet.";
}

Status HandleFusion(HloInstruction* fusion) {
  LOG(FATAL) << "Not implemented yet.";
}

Status HandleCustomCall(HloInstruction* custom_call) {
  LOG(FATAL) << "Not implemented yet.";
}

Status FinishVisit(HloInstruction* root) {
  LOG(FATAL) << "Not implemented yet.";
}
}  // namespace gpu
}  // namespace xla
