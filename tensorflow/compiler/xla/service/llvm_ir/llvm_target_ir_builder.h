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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_TARGET_IR_BUILDER_H__
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_TARGET_IR_BUILDER_H__

#include "llvm_target_features.h"

#include "llvm/IR/IRBuilder.h"

namespace xla {
namespace llvm_ir {
// This class encapsulates the IR builder and llvm target information 
class LLVMTargetIRBuilder {
 public:
  LLVMTargetIRBuilder(llvm::IRBuilder<>* b,
                      LLVMTargetFeatures* llvm_target_features)
      : b_(b), llvm_target_features_(llvm_target_features) {}

  llvm::IRBuilder<>* builder() { return b_; }

  LLVMTargetFeatures* GetTargetMachineFeatures() {
    return llvm_target_features_;
  };
private:
  llvm::IRBuilder<>* b_;
  LLVMTargetFeatures* llvm_target_features_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_IR_BUILDER_H_
