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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_HELPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_HELPER_H_

#include <string>
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

namespace xla {
namespace llvm_ir {
// Enmerations to get target specific intrinsics or function calls
enum TargetIntrinsicID {
  kShfl_down_f32,
  kShfl_down_i32,
  kThread_id_x,
  kThread_id_y,
  kThread_id_z,
  kBlock_id_x,
  kBlock_id_y,
  kBlock_id_z,
  kBarrier_id,
  kLast_id
};

struct TargetIntrinsics {
  llvm::Intrinsic::ID nvptx_intrinsic;
  llvm::Intrinsic::ID amdgpu_intrinsic;
};

typedef struct TargetIntrinsics GPUIntrinsics; 

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_HELPER_H_
